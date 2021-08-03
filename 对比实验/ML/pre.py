# from __future__ import absolute_import
# from __future__ import print_function
import os
import socket
import sys
from constants import PRETTY_NAMES
import numpy as np
from asn1crypto import x509


from cal import *


import dpkt

global contact
contact = {}


need_more_parse = True  # 需要更多TLS信息
need_more_certificate = True
need_http = False
need_matrix = False

class features(object):
    def __init__(self):
        self.ip_src = ''  # 目的ip地址
        self.ip_dst = ''  # 源ip地址
        self.dport = 0  # 源端口号
        self.sport = 0  # 目的端口号
        self.pack_num = 0  # 包数量
        self.connection_num = 0  # 流数目
        self.num_src = 0  # 源包数目
        self.num_dst = 0    # 目的包数目
        self.num_ratio = 0  # 上下行流量比
        self.size_src = 0   # 源总包大小
        self.size_dst = 0   # 目的总包大小
        self.ratio = 0  # 上下行包大小比
        self.by_s = 0   # 每秒字节传输速度
        self.pk_s = 0   # 每秒包传输速度
        self.time = 0  # 整体持续时间
        self.time_sequence = []  # 时间序列
        self.max_time = 0  # 最大间隔时间
        self.min_time = 0  # 最小间隔时间
        self.mean_time = 0  # 平均间隔时间
        self.std_time = 0  # 均差间隔时间
        self.time_src_sequence = []  # 源时间间隔序列
        self.max_time_src = 0  # 最大源时间间隔
        self.min_time_src = 0  # 最小源时间间隔
        self.mean_time_src = 0  # 平均源时间间隔
        self.std_time_src = 0  # 均差源时间间隔
        self.time_dst_sequence = []  # 目的时间间隔序列
        self.max_time_dst = 0  # 最大目的时间间隔
        self.min_time_dst = 0  # 最小目的时间间隔
        self.mean_time_dst = 0  # 平均目的时间间隔
        self.std_time_dst = 0  # 均差目的时间间隔
        self.packetsize_src_sequence = []  # 源包大小序列
        self.max_packetsize_src = 0  # 最大源包大小
        self.min_packetsize_src = 0  # 最小源包大小
        self.mean_packetsize_src = 0  # 平均源包大小
        self.std_packetsize_src = 0  # 均差源包大小
        self.packetsize_dst_sequence = []  # 目的包大小序列
        self.max_packetsize_dst = 0  # 最大目的包大小
        self.min_packetsize_dst = 0  # 最小目的包大小
        self.mean_packetsize_dst = 0  # 均值目的包大小
        self.std_packetsize_dst = 0  # 均差目的包大小
        self.packetsize_flow_sequence = []  # 流大小序列
        self.max_packetsize_flow = 0  # 最大流大小
        self.min_packetsize_flow = 0  # 最小流大小
        self.mean_packetsize_flow = 0  # 平均流大小
        self.std_packetsize_flow = 0  # 均差流大小
        self.time_flow_sequence = []  # 流时间序列
        self.max_time_flow = 0  # 最大流时间
        self.min_time_flow = 0  # 最小流时间
        self.mean_time_flow = 0  # 平均流时间
        self.std_time_flow = 0  # 均差流时间
        self.packetsize_size = 0  # # 平均包大小
        self.packetsize_packet_sequence = []  # 包大小序列
        self.max_packetsize_packet = 0  # 最大包大小
        self.min_packetsize_packet = 0  # 最小包大小
        self.mean_packetsize_packet = 0  # 平均包大小
        self.std_packetsize_packet = 0  # 均差包大小
        self.sequence = []  # 自TLS开始的有向序列
        self.num = 0  # 数据流数量
        self.cipher_num = 0  # 加密组件长度
        self.cipher_support = []  # 加密支持组件序列
        self.cipher_support_num = 0  # 加密支持组件编码
        self.cipher = 0  # 加密组件
        self.name = ''  # pacp包名称
        self.cipher_content = [0 for i in range(256)]  # 加密内容里各字节出现次数
        self.bit_fre_ratio = 0  # 加密内容位中0出现次数
        self.flag = False
        self.fin = 0  # 标志位Fin的数量
        self.syn = 0  # 标志位Syn的数量
        self.rst = 0  # 标志位RST的数量
        self.ack = 0  # 标志位ACK的数量
        self.urg = 0  # 标志位URG的数量
        self.psh = 0  # 标志位PSH的数量
        self.ece = 0  # 标志位ECE的数量
        self.cwe = 0  # 标志位CWE的数量
        self.transition_matrix = np.zeros((15, 15), dtype=int)  # 马尔可夫转移矩阵
        self.lable = ''


    def tolist(self):
        time = round(self.time)
        # ip_src = int(self.ip_src.replace('.', ''))
        self.packetsize_size = round(self.packetsize_size / self.pack_num)
        self.max_time, self.min_time, self.mean_time, self.std_time = cal(self.time_sequence)
        self.max_packetsize_flow, self.min_packetsize_flow, self.mean_packetsize_flow, self.std_packetsize_flow = cal(
            self.packetsize_flow_sequence)
        self.max_time_flow, self.min_time_flow, self.mean_time_flow, self.std_time_flow = cal(self.time_flow_sequence)
        self.time_src_sequence = cal_seq(self.time_src_sequence)
        self.time_dst_sequence = cal_seq(self.time_dst_sequence)
        self.max_time_src, self.min_time_src, self.mean_time_src, self.std_time_src = cal(self.time_src_sequence)
        self.max_time_dst, self.min_time_dst, self.mean_time_dst, self.std_time_dst = cal(self.time_dst_sequence)
        self.max_packetsize_src, self.min_packetsize_src, self.mean_packetsize_src, self.std_packetsize_src = cal(
            self.packetsize_src_sequence)
        self.max_packetsize_dst, self.min_packetsize_dst, self.mean_packetsize_dst, self.std_packetsize_dst = cal(
            self.packetsize_dst_sequence)
        self.max_packetsize_packet, self.min_packetsize_packet, self.mean_packetsize_packet, self.std_packetsize_packet = cal(
            self.packetsize_packet_sequence)
        self.cipher_support_num = cal_hex(self.cipher_support)
        self.bit_fre_ratio = round(cal_ratio(self.cipher_content), 4)
        if need_matrix:
            self.transition_matrix = cal_matrix(feature.packetsize_packet_sequence)
        self.num_ratio = cal_div(self.num_src, self.num_dst)
        self.size_ratio = cal_div(self.size_src, self.num_dst)
        self.by_s = cal_div(self.packetsize_size,self.time)
        self.pk_s = cal_div(self.pack_num, self.time)
        return [self.pack_num, time, self.connection_num, self.cipher_num, self.packetsize_size,
                self.max_time, self.min_time, self.mean_time, self.std_time, 
                self.max_time_src, self.min_time_src, self.mean_time_src, self.std_time_src,
                self.max_time_dst, self.min_time_dst, self.mean_time_dst, self.std_time_dst, 
                self.max_time_flow, self.min_time_flow, self.mean_time_flow, self.std_time_flow,
                self.max_packetsize_packet, self.mean_packetsize_packet, self.std_packetsize_packet, 
                self.max_packetsize_src, self.mean_packetsize_src, self.std_packetsize_src,
                self.max_packetsize_dst, self.mean_packetsize_dst, self.std_packetsize_dst,
                self.max_packetsize_flow, self.min_packetsize_flow, self.mean_packetsize_flow, self.std_packetsize_flow,
                self.bit_fre_ratio,
                self.lable
                ]
        """
        self.pack_num = 0  # 包数量
        time # 整体持续时间
        self.connection_num = 0  # 流数目（加密链接次数）
       
     
        self.max_time = 0  # 最大间隔时间
        self.min_time = 0  # 最小间隔时间
        self.mean_time = 0  # 平均间隔时间
        self.std_time = 0  # 均差间隔时间

        self.max_time_src = 0  # 最大源时间间隔
        self.min_time_src = 0  # 最小源时间间隔
        self.mean_time_src = 0  # 平均源时间间隔
        self.std_time_src = 0  # 均差源时间间隔
    
        self.max_time_dst = 0  # 最大目的时间间隔
        self.min_time_dst = 0  # 最小目的时间间隔
        self.mean_time_dst = 0  # 平均目的时间间隔
        self.std_time_dst = 0  # 均差目的时间间隔

        self.max_time_flow = 0  # 最大流时间
        self.min_time_flow = 0  # 最小流时间
        self.mean_time_flow = 0  # 平均流时间
        self.std_time_flow = 0  # 均差流时间

        self.max_packetsize_src = 0  # 最大源包大小
        self.min_packetsize_src = 0  # 最小源包大小
        self.mean_packetsize_src = 0  # 平均源包大小
        self.std_packetsize_src = 0  # 均差源包大小
 
        self.max_packetsize_dst = 0  # 最大目的包大小
        self.min_packetsize_dst = 0  # 最小目的包大小
        self.mean_packetsize_dst = 0  # 均值目的包大小
        self.std_packetsize_dst = 0  # 均差目的包大小

        # 计算上行流和下行流
        self.max_packetsize_flow = 0  # 最大流大小
        self.min_packetsize_flow = 0  # 最小流大小
        self.mean_packetsize_flow = 0  # 平均流大小
        self.std_packetsize_flow = 0  # 均差流大小
      
        self.max_packetsize_packet = 0  # 最大包大小
        self.min_packetsize_packet = 0  # 最小包大小
        self.mean_packetsize_packet = 0  # 平均包大小
        self.std_packetsize_packet = 0  # 均差包大小

        self.fin = 0  # 标志位Fin的数量
        self.syn = 0  # 标志位Syn的数量
        self.rst = 0  # 标志位RST的数量
        self.ack = 0  # 标志位ACK的数量
        self.urg = 0  # 标志位URG的数量
        self.psh = 0  # 标志位PSH的数量
        self.ece = 0  # 标志位ECE的数量
        self.cwe = 0  # 标志位CWE的数量

        self.num_src = 0  # 源包数目
        self.num_dst = 0    # 目的包数目
        self.num_ratio = 0  # 上下行流量比

        self.size_src = 0   # 源总包大小
        self.size_dst = 0   # 目的总包大小
        self.ratio = 0  # 上下行包大小比
   
        self.by_s = 0   # 每秒字节传输速度
        self.pk_s = 0   # 每秒包传输速度
        """
        
        #  dport
        # return [self.pack_num, time, self.dport, self.connection_num, ip_src, self.cipher_num, self.packetsize_size,
        #         self.max_time, self.min_time, self.mean_time, self.std_time, self.max_time_src, self.min_time_src, self.mean_time_src, self.std_time_src,
        #         self.max_time_dst, self.min_time_dst, self.mean_time_dst, self.std_time_dst, self.max_time_flow, self.min_time_flow, self.mean_time_flow, self.std_time_flow,
        #         self.max_packetsize_packet,  self.mean_packetsize_packet, self.std_packetsize_packet,
        #         self.max_packetsize_src,  self.mean_packetsize_src, self.std_packetsize_src,
        #         self.max_packetsize_dst, self.mean_packetsize_dst, self.std_packetsize_dst,
        #         self.max_packetsize_flow, self.min_packetsize_flow, self.mean_packetsize_flow, self.std_packetsize_flow,
        #         self.bit_fre_ratio
        #         ]


class feature_type:
    def __init__(self, num, flow_size, flow_starttime, flow_endtime):
        self.num = num
        self.flow_size = flow_size
        self.flow_duration = 0
        self.flow_starttime = flow_starttime
        self.flow_endtime = flow_endtime


def pretty_name(name_type, name_value):
    """Returns the pretty name for type name_type."""
    if name_type in PRETTY_NAMES:
        if name_value in PRETTY_NAMES[name_type]:
            name_value = PRETTY_NAMES[name_type][name_value]
        else:
            name_value = '{0}: unknown value {1}'.format(name_value, name_type)
    else:
        name_value = 'unknown type: {0}'.format(name_type)
    return name_value


def analyze_packet(timestamp, packet, nth):
    """
    Main analysis loop for pcap.
    """

    eth = dpkt.ethernet.Ethernet(packet)
    if isinstance(eth.data, dpkt.ip.IP):
        parse_ip_packet(eth, nth, timestamp)


def parse_ip_packet(eth, nth, timestamp):
    """
    Parses IP packet.
    """
    ip = eth.data
    tcp_data = ip.data
    sys.stdout.flush()
    size = len(eth)  # 包大小
    feature.packetsize_packet_sequence.append(size)
    payload = len(ip.data.data)  # 有效负载大小
    rest_load = None
    if isinstance(tcp_data, dpkt.tcp.TCP):
        feature.fin += 1 if cal_fin(tcp_data.flags) else 0
        feature.syn += 1 if cal_syn(tcp_data.flags) else 0
        feature.rst += 1 if cal_rst(tcp_data.flags) else 0
        feature.psh += 1 if cal_psh(tcp_data.flags) else 0
        feature.ack += 1 if cal_ack(tcp_data.flags) else 0
        feature.urg += 1 if cal_urg(tcp_data.flags) else 0
        feature.cwe += 1 if cal_cwe(tcp_data.flags) else 0
        feature.ece += 1 if cal_ece(tcp_data.flags) else 0

    # 提取 ip地址、端口号
    if nth == 1:
        feature.ip_src = socket.inet_ntoa(ip.src)
        feature.ip_dst = socket.inet_ntoa(ip.dst)
        try:
            feature.sport = int(ip.data.sport)
            feature.dport = int(ip.data.dport)
        except AttributeError as exception:
            feature.sport = 0
            feature.dport = 0
    if socket.inet_ntoa(ip.src) == feature.ip_src:
        feature.time_src_sequence.append(timestamp)
        feature.packetsize_src_sequence.append(size)
        feature.num_src += 1
        feature.size_src += size
    else:
        feature.time_dst_sequence.append(timestamp)
        feature.packetsize_dst_sequence.append(size)
        feature.num_dst += 1
        feature.size_dst += size
    try:
        flag = socket.inet_ntoa(ip.src) + ' ' + socket.inet_ntoa(ip.dst) + ' ' + str(ip.data.dport) + ' ' + str(
        ip.data.sport)
        flag_1 = socket.inet_ntoa(ip.dst) + ' ' + socket.inet_ntoa(ip.src) + ' ' + str(ip.data.sport) + ' ' + str(
        ip.data.dport)
    except AttributeError as exception:
        flag = socket.inet_ntoa(ip.src) + ' ' + socket.inet_ntoa(ip.dst)
        flag_1 = socket.inet_ntoa(ip.dst) + ' ' + socket.inet_ntoa(ip.src)

    if contact.__contains__(flag):
        contact[flag].num += 1
        contact[flag].flow_endtime = timestamp
        contact[flag].flow_size += size
    # elif contact.__contains__(flag_1):
    #     contact[flag_1].num += 1
    #     contact[flag_1].flow_endtime = timestamp
    #     contact[flag_1].flow_size += size
    else:
        tem = feature_type(0, size, timestamp, timestamp)
        contact[flag] = tem
    feature.packetsize_size += size

    if isinstance(ip.data, dpkt.tcp.TCP) and payload:
        rest_load = parse_tcp_packet(ip, nth, timestamp)
        for key in ip.data.data:
            feature.cipher_content[key] += 1
        if socket.inet_ntoa(ip.src) == feature.ip_dst:
            dir = 1
        else:
            dir = -1
        Dir = dir * payload
        if len(feature.sequence) < 20:
            feature.sequence.append(Dir)

    if need_more_certificate:
        class flow_type:
            def __init__(self, seq, data):
                self.seq = seq
                self.data = data
                self.sequence = []

        # 设置flow记录流的各条记录，以解决tcp reasembeld segement
        flow_flag = socket.inet_ntoa(ip.src) + '->' + socket.inet_ntoa(ip.dst)
        flow_flag1 = socket.inet_ntoa(ip.dst) + '->' + socket.inet_ntoa(ip.src)
        # 存在udp 没有seq和ack
        try:
            seq = ip.data.seq
            ack = ip.data.ack
        except AttributeError as exception:
            seq = 0
            ack = 0

        if rest_load != None:
            if len(rest_load):
                data = rest_load
            else:
                data = ip.data.data
            # print(nth, data, len(data))
        else:
            data = ip.data.data
        if len(ip.data.data):
            if flow_flag1 in flow.keys():
                if ack > flow[flow_flag1].seq:
                    if len(flow[flow_flag1].data) != 0:
                        tem = flow[flow_flag1].data
                        if tem[0] in {20, 21, 22}:
                            parse_tls_records(ip, tem, nth)
                        # print(nth, flow[flow_flag1].data)
                    flow[flow_flag1].data = bytes(0)

        if flow_flag not in flow.keys():
            if data != bytes(0):
                if data[0] in {20, 21, 22}:
                    flow[flow_flag] = flow_type(seq, data)
                    flow[flow_flag].sequence.append(data)
        else:
            if flow[flow_flag].seq != seq:
                flow[flow_flag].seq = seq
                if data not in flow[flow_flag].sequence:
                    flow[flow_flag].data += data
                    flow[flow_flag].sequence.append(data)
            else:
                flow[flow_flag].data = data
                flow[flow_flag].sequence.clear()
                flow[flow_flag].sequence.append(data)

    # print(nth, socket.inet_ntoa(ip.src) + '->' + socket.inet_ntoa(ip.dst), seq, ack)


def parse_tcp_packet(ip, nth, timestamp):
    """
    Parses TCP packet.
    """
    rest_load = None
    tcp_data = ip.data
    stream = ip.data.data
    # http 协议
    # if need_http:
    #     class flow_type:
    #         def __init__(self, seq, data):
    #             self.seq = seq
    #             self.data = data
    #             self.sequence = []
    #
    #         # 设置flow记录流的各条记录，以解决tcp reasembeld segement
    #
    #     flow_flag = socket.inet_ntoa(ip.src) + '->' + socket.inet_ntoa(ip.dst)
    #     data = ip.data.data
    #     if flow_flag in flow.keys():
    #         data = flow[flow_flag].data + ip.data.data
    #     else:
    #         flow[flow_flag] = flow_type(0, ip.data.data)
    #     print(nth,data)
    #     try:
    #         request = dpkt.http.Response(data)
    #         print('HTTP request: %s\n' % repr(request))
    #     except(dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
    #         print("1")
    #         pass


    # ssl flow
    #  提取标志位

    # if cal_psh(tcp_data.flags):
    #     feature.psh += 1
    # if cal_urg(tcp_data.data.flags):
    #     feature.urg += 1
    if need_more_parse:
        if (stream[0]) in {20, 21, 22, 23, 128, 25}:
            if (stream[0]) in {20, 21, 22}:
                rest_load = parse_tls_records(ip, stream, nth)
            if (stream[0]) == 128:  # sslv2 client hello
                # feature.flag = True
                try:
                    length = int(stream[6])
                except:
                    length = 0
                if stream[2] == 1:
                    print(nth)
                    feature.cipher_num = max(length, feature.cipher_num)
                    tem = stream[7] + 11
                    i = 0
                    while i < length:
                        cipher = 0
                        if tem + i + 2 < len(stream):
                            cipher = stream[tem + i + 2] + stream[tem + i + 1] * 256 + stream[tem + i] * 256 * 256

                        if cipher not in feature.cipher_support:
                            feature.cipher_support.append(cipher)
                        i += 3
                    # print(nth, stream[6])
            if (stream[0]) == 25:
                rest_load = parse_tls_records(ip, stream, nth)
    return rest_load


def parse_tls_records(ip, stream, nth):
    """
    Parses TLS Records.
    """

    try:
        records, bytes_used = dpkt.ssl.tls_multi_factory(stream)
    except dpkt.ssl.SSL3Exception as exception:
        return
    # connection = '{0}:{1}-{2}:{3}'.format(socket.inet_ntoa(ip.src),
    #                                       ip.data.sport,
    #                                       socket.inet_ntoa(ip.dst),
    #                                       ip.data.dport)
    n = 0

    for record in records:
        # print(nth, record.version)
        record_type = pretty_name('tls_record', record.type)
        if record_type == 'handshake':
            handshake_type = ord(record.data[:1])
            packetsize = record.data
            if handshake_type == 2:  # server hello
                feature.connection_num += 1
                feature.cipher = (record.data[-2] + record.data[-3] * 256)
            if handshake_type == 11:  # certificate
                len_cer = int.from_bytes(record.data[4:7], byteorder='big')  # 转换字节流为十进制
                data = record.data[7:]
                tem = 0
                a = []
                a = parse_tls_certs(nth, record.data, record.length)
                while len(data):
                    len_cer_tem = int.from_bytes(data[0:3], byteorder='big')
                    certificate = data[3:len_cer_tem + 3]
                    data = data[len_cer_tem + 3:]
            if n == 0:
                if handshake_type == 1:  # sslv3 tlsv1 client hello
                    # feature.flag = True
                    length = int(record.data[40 + record.data[38]])
                    feature.cipher_num = max(length, feature.cipher_num)
                    tem = 40 + record.data[38] + 1
                    i = 0
                    while i < length:
                        cipher = record.data[tem + i] * 256 + record.data[tem + i + 1]
                        if cipher not in feature.cipher_support:
                            feature.cipher_support.append(cipher)
                        i += 2

                    # print(nth, record.data[40])
        n += 1
        sys.stdout.flush()
    # ressembled tcp segments
    load = stream[bytes_used:]
    return load


def parse_tls_certs(nth, data, record_length):
    """
    Parses TLS Handshake message contained in data according to their type.
    """
    ans = []
    handshake_type = ord(data[:1])  # 握手类型
    if handshake_type == 4:
        print('[#] New Session Ticket is not implemented yet')
        return ans

    buffers = data[0:]
    try:
        handshake = dpkt.ssl.TLSHandshake(buffers)
    except dpkt.ssl.SSL3Exception as exception:
        pass
        # print('exception while parsing TLS handshake record: {0}'.format(exception))
    except dpkt.dpkt.NeedData as exception:
        pass
        # print('exception while parsing TLS handshake record: {0}'.format(exception))
    try:
        ch = handshake.data
    except UnboundLocalError as exception:
        pass
    else:
        if handshake.type == 11:  # TLS Certificate
            # ssl_servers_with_handshake.add(client)
            hd_data = handshake.data
            assert isinstance(hd_data, dpkt.ssl.TLSCertificate)
            certs = []
            # print(dir(hd))
            for i in range(len(hd_data.certificates)):
                # print("hd.certificates[i]:", hd_data.certificates[i])
                cert = x509.Certificate.load(hd_data.certificates[i])
                # print(cert.public_key)
                # self_signed = cert.self_signed  # 是否自签名
                before = cert.not_valid_before
                after = cert.not_valid_after
                # print(before, after)
                # print("subject:", cert.subject)
                # print("issuer:", cert.issuer)
                # print("issuer_serial:", cert.issuer_serial)
                sha = cert.sha256_fingerprint.replace(" ", "")
                # print(sha)
                certs.append(sha)
            ans += certs
        # print('exception while parsing TLS handshake record: {0}'.format(exception))

    return ans


def read_file(filename):
    try:
        with open(filename, 'rb') as f:
            capture = dpkt.pcap.Reader(f)
            nth = 1
            time = 0
            global feature
            feature = features()
            global flow
            flow = {}
            # feature.ip_src = filename.replace('.pcap', '').replace(base_dir, '')
            contact.clear()
            seq = []
            for timestamp, packet in capture:
                analyze_packet(timestamp, packet, nth)
                if nth == 1:
                    flag = timestamp
                time = timestamp - flag
                nth += 1
                seq.append(time)
            feature.time_sequence = cal_seq(seq)
            for key in contact:
                contact[key].duration = contact[key].flow_endtime - contact[key].flow_starttime
                feature.packetsize_flow_sequence.append(contact[key].flow_size)
                feature.time_flow_sequence.append(contact[key].duration)
            # flow 剩余解析certificate
            if need_more_certificate:
                for key, value in flow.items():
                    if len(value.data) != 0:
                        tem = value.data
                        if tem[0] in {20, 21, 22}:
                            parse_tls_records(tem, tem, nth)
            feature.pack_num = nth
            feature.time = time
            while len(feature.sequence) < 20:
                feature.sequence.append(0)
            f.close()
    except IOError:
        print('could not parse {0}'.format(filename))


def pre_pcap(base_dir, type):
    dataset = []
    i = 0
    print(base_dir)
    for filename in os.listdir(base_dir):
        i += 1
        if i % 10000 == 0:
            print(i)
        read_file(base_dir + "/" +filename)
        feature.lable = type
        # feature.name = filename.replace('.pcap', '')
        dataset.append(feature.tolist())
    print("data collect end")
    return dataset
    # for file in os.listdir(base_dir):
    #     dir = base_dir + file + '/'
    #     print(file)
    #     print(dir)
    #     i = 0
    #     for filename in os.listdir(dir):
    #         # i += 1
    #         # if i == 10:
    #         #     break
    #         read_file(dir + filename)
    #         feature.name = filename.replace('.pcap', '')
    #         feature.lable = file
    #         dataset.append(feature.tolist())
    # print("data collect end")
    # return dataset




def main():
    print("begin")
    dataset = []
    i = 0
    base_dir = "data/eta_1/train/black/"
    for filename in os.listdir(base_dir):
        i += 1
        if i == 2:
            break
        # filename = "192.168.133.165.pcap"
        # filename = "192.168.71.170.pcap"
        # filename = "192.168.0.233.pcap"
        # filename = "192.168.114.127.pcap"
        # filename = "192.168.193.239.pcap"
        # filename = "192.168.0.233.pcap"
        # filename = "192.168.253.95.pcap"
        # filename = "192.168.163.190.pcap"
        # filename = "192.168.168.108.pcap" # 并行重传
        # filename =  "192.168.225.157.pcap" # udp

        filename = "192.168.108.126.pcap"
        print(filename)
        read_file(base_dir + filename)
        feature.name = filename.replace('.pcap', '')
        print(feature.cipher_support)
        dataset.append(feature.tolist())
    print("end")


if __name__ == "__main__":
    main()
