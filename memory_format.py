a = """%Agriadapt unet: 
%data: 6291456
%+net: 9879040
%+backwardmax: 2534039040
%+net perfconv (all)): 9986560
%+backwardmax 1,1: 2530764288
%+backwardmax 2,2: 2530764288
%+backwardmax 3,3: 2530764288
%+net dau: 9879040
%+backwardmax 1,1: 1797473792
%+backwardmax 2,2: 1864587776
%+backwardmax 3,3: 1679108608
%_Our unet:
%data: 6291456
%+net: 15483392
%bmax: 1325407232
%+net perf: 15508480
%bmax 1,1: 1330565120
%bmax 2,2: 1330565120
%bmax 3,3: 1330565120
%+net dau: 14783488
%bmax 1,1: 1186461696
%bmax 2,2: 972667904
%bmax 3,3: 882339328
%_resnet18:
%data: 24576
%+net: 109720576
%bmax: 129526784
%+net perf: 109982720
%bmax 1,1: 129788928
%bmax 2,2: 129788928
%bmax 3,3: 129788928
%+net dau: 154067456
%bmax 1,1: 173431296
%bmax 2,2: 173579776
%bmax 3,3: 173538816
%_mobnetv2:
%data: 24576
%+net: 35205120
%bmax: 42194944
%+net perf: 44583424
%bmax 1,1: 51573248
%bmax 2,2: 51573248
%bmax 3,3: 51573248
%+net_dau: 35054592
%bmax 1,1: 39831040
%bmax 2,2: 40003072
%bmax 3,3: 39948800
%_mobnetv3small:
%data: 24576
%+net: 29426688
%bmax: 32581120
%+netperf: 29426688
%bmax 1,1: 32581120
%bmax 1,1: 32581120
%bmax 1,1: 32581120
%+netdau: 29365248
%bmax 1,1: 32070144
%bmax 2,2: 32105984
%bmax 3,3: 32097792"""

#These above are measured max byte allocation values during forward/backward pass (from model_resources,
# but measured manually, as allocation values sometimes change during programmatic measuring)
if __name__ == "__main__":
    for group in a.split("%_"):
        lines = group.split("\n")
        name = lines[0]
        data = int(lines[1].split(" ")[-1])
        base_net = int(lines[2].split(" ")[-1]) - data
        net_perf = int(lines[4].split(" ")[-1]) - data
        net_dau = int(lines[8].split(" ")[-1]) - data

        base_back = int(lines[3].split(" ")[-1]) - data - base_net

        base_perf1 = int(lines[5].split(" ")[-1]) - data - net_perf
        base_perf2 = int(lines[6].split(" ")[-1]) - data - net_perf
        base_perf3 = int(lines[7].split(" ")[-1]) - data - net_perf

        base_dau1 = int(lines[9].split(" ")[-1]) - data - net_dau
        base_dau2 = int(lines[10].split(" ")[-1]) - data - net_dau
        base_dau3 = int(lines[11].split(" ")[-1]) - data - net_dau
        print(name)
        print(int(1000 * (net_perf/base_net))/10, "\\%, dau:", int(1000*(net_dau/base_net)/10), "%")
        print("Base:100%,  1x1 perf:", int(1000 * (base_perf1/base_back))/10, "%, 1x1 dau:", int(1000*(base_dau1/base_back)/10), "%")
        print("Base 2x2:100%, 2x2 perf:", int(1000 * (base_perf2/base_back))/10, "%, 2x2 dau:", int(1000*(base_dau2/base_back)/10), "%")
        print("Base 3x3:100%, 3x3 perf:", int(1000 * (base_perf3/base_back))/10, "%, 3x3 dau:", int(1000*(base_dau3/base_back)/10), "%")
        print("\n--------------\n")
    ...