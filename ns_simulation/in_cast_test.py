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
    return 0.1


def packet_size():
    """The packets have a constant size of 1024 bytes."""
    return 512


def delay_dist():
    """Network wires experience a constant propagation delay of 0.1 seconds."""
    return 0.1


# 1 * receiver, in_cast_degree * sender, 1 * switch
in_cast_degree = 2

env = simpy.Environment()

switch = SimplePacketSwitch(
    env,
    nports=in_cast_degree + 1,
    port_rate=16384,  # in bits/second
    buffer_size=5,  # in packets
    debug=True,
)
# switch 端口连接：0端口接“汇聚节点”，N端口接sender N ...
# Sender id 从1开始

receiver = TCPSink(env, rec_waits=True, debug=True)

flow_bank = []
sender_bank = []
fib = {}

for sender_id in range(1, in_cast_degree+1):
    # New object
    new_flow = Flow(fid=sender_id, src="flow " + str(sender_id), dst="flow " + str(sender_id), finish_time=10,
                    arrival_dist=packet_arrival, size_dist=packet_size, )
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


switch.demux.fib = fib
# receiver
switch.demux.outs[0] = receiver
receiver.out = switch

env.run(until=1)

