import simpy

from ns.flow.cc import TCPReno
from ns.flow.cubic import TCPCubic
from ns.flow.flow import AppType, Flow
from ns.packet.tcp_generator import TCPPacketGenerator
from ns.packet.tcp_sink import TCPSink
from ns.port.wire import Wire
from ns.switch.switch import SimplePacketSwitch


def packet_arrival():
    """Packets arrive with a constant interval of 0.1 seconds."""
    # 10G
    return 0.1


def packet_size():
    """The packets have a constant size of 1024 bytes."""
    return 512


def delay_dist():
    """Network wires experience a constant propagation delay of 0.1 seconds."""
    # 10m
    return 0.1


# 1 * receiver, in_cast_degree * sender, 1 * switch
in_cast_degree = 8
port_rate_bps = 16384  # bps
# buffer_size=1000 # in packets
flow_size = 100000  # in bytes

env = simpy.Environment()

switch = SimplePacketSwitch(
    env,
    nports=in_cast_degree + 1,
    port_rate=port_rate_bps,  # in bits/second
    buffer_size=1000,  # in packets
    debug=True,
)
# switch 端口连接：0端口接“汇聚节点”，N端口接sender N ...
# Sender id 从1开始

# agg_node 端口连接：0端口接switch，N端口接sink N ...
agg_node = SimplePacketSwitch(
    env,
    nports=in_cast_degree + 1,
    port_rate=port_rate_bps,  # in bits/second
    buffer_size=1000,  # in packets
    debug=True,
)

flow_bank = []
sender_bank = []
# fib for switch
fib = {}
# fib for agg.node
fib_an = {}

for sender_id in range(1, in_cast_degree+1):
    # New object
    new_flow = Flow(fid=sender_id, src="flow " + str(sender_id), dst="flow " + str(sender_id), finish_time=100000,
                    arrival_dist=packet_arrival, size_dist=packet_size, size=flow_size)
    new_sender = TCPPacketGenerator(env, flow=new_flow, cc=TCPReno(), element_id=new_flow.src, debug=True)
    new_wire1_downstream = Wire(env, delay_dist)
    new_wire1_upstream = Wire(env, delay_dist)

    flow_bank.append(new_flow)
    sender_bank.append(new_sender)

    # update Flow information base: {flow_id: output switch_port_id}
    fib[sender_id] = 0  # to receiver
    fib[sender_id + 10000] = sender_id  # ACK to sender i; why 10000?: ACK.flow_id=packet.flow_id + 10000

    # connecting object: on the side of senders
    new_sender.out = new_wire1_downstream
    new_wire1_downstream.out = switch
    switch.demux.outs[sender_id].out = new_wire1_upstream
    new_wire1_upstream.out = new_sender

    fib_an[sender_id] = sender_id
    fib_an[sender_id + 10000] = 0
    new_receiver = TCPSink(env, rec_waits=True, debug=True)
    agg_node.demux.outs[sender_id].out = new_receiver
    new_receiver.out = agg_node


# connecting object: on the side of receiver
wire2_downstream = Wire(env, delay_dist)
wire2_upstream = Wire(env, delay_dist)

switch.demux.fib = fib
switch.demux.outs[0].out = wire2_downstream
wire2_downstream.out = agg_node

agg_node.demux.fib = fib_an
agg_node.demux.outs[0].out = wire2_upstream
wire2_upstream.out = switch


env.run(until=1000)




