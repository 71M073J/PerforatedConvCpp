import os
import json

import numpy as np

architectures = [
        [[(..., "resnet18"), (..., "mobnetv2"), (..., "mobnetv3s")],
         ["cifar", "ucihar"], [32]],
        # "cinic" takes too long to run, ~45sec per epoch compared to ~9 for cifar ,so it would be about 2 hour training per config, maybe later
        [[(..., "unet_agri"), (..., "unet_custom")], ["agri"], [128, 256, 512]]
    ]


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def type_convert(typ):
    if typ == "dau":
        return "Upscale after activation"
    if typ == "perf":
        return "Normal perforation"
    return "No perforation"

output = {}

prefix = "allTests_last"
for version in architectures:  # classigication, segmetnationg
    for dataset in version[1]:
        if dataset == "agri":
            lr = 0.5
            max_epochs = 10
            batch_size = 32
        elif dataset == "ucihar":
            max_epochs = 10
            batch_size = 32
            lr = 0.01
        else:
            max_epochs = 10
            batch_size = 32
            lr = 0.01

        for model, modelname in version[0]:
            for img in version[2]:
                # model = UNet
                # img = 128
                # dataset = "agri"
                img_res = (img, img)
                in_size = (2, 3, img, img)
                if img == 128:
                    batch_size = 32
                elif img == 256:
                    batch_size = 16
                elif img == 512:
                    batch_size = 4

                alreadyNoPerf = False
                for perforation in (None, 2, 3, "random", "2by2_equivalent"):
                    perf = (perforation, perforation)
                    if perforation is None:
                        if alreadyNoPerf:
                            continue
                        else:
                            alreadyNoPerf = True

                    for perf_type in ["perf", "dau"]:
                        if perforation is None:
                            if perf_type == "dau":
                                continue
                            else:
                                perf_type = None
                        name = f"{modelname}_{dataset}_{img}_{perforation}_{perf_type}"

                        graphname = f"{modelname}_{img}_{dataset}_{perf_type}"

                        curr_file = f"{name}"


                        run_name = f"{curr_file}"
                        if not os.path.exists(f"./{prefix}/"):
                            os.mkdir(f"./{prefix}")
                        if not os.path.exists(f"./{prefix}/imgs"):
                            os.mkdir(f"./{prefix}/imgs")
                        for extra in ["", "/cpu_old"]:
                            with open(f"{prefix}{extra}/{run_name}_best.txt") as data:
                                line = data.readline()
                                pf = "Training perforation: " + str(perforation)
                                for evaluation in replace_all(line, {"[":"", "]":"", "Validation ":"", "((":"(", "))":")"}).split("', '"):
                                    if extra == "":
                                        if "acc" in evaluation:
                                            eval_mode = evaluation.split(":")[0].split(" (")[1][0:4]
                                            eval_mode = eval_mode if eval_mode != "None" else "Same as training"
                                            eval_mode = "Evaluation perforation: " + eval_mode
                                            acc = evaluation.split(":")[1] + "%"
                                            img = str(img)
                                            #perforation = #str(perforation)
                                            if modelname not in output:
                                                output[modelname] = {dataset:{type_convert(perf_type):{img:{pf:{eval_mode:acc}}}}}
                                            elif dataset not in output[modelname]:
                                                output[modelname][dataset] = {type_convert(perf_type):{img:{pf:{eval_mode:acc}}}}
                                            elif type_convert(perf_type) not in output[modelname][dataset]:
                                                output[modelname][dataset][type_convert(perf_type)] = {img:{pf:{eval_mode:acc}}}
                                            elif img not in output[modelname][dataset][type_convert(perf_type)]:
                                                output[modelname][dataset][type_convert(perf_type)][img] = {pf:{eval_mode:acc}}
                                            elif pf not in output[modelname][dataset][type_convert(perf_type)][img]:
                                                output[modelname][dataset][type_convert(perf_type)][img][pf] = {eval_mode:acc}
                                            else:
                                                output[modelname][dataset][type_convert(perf_type)][img][pf][eval_mode] = acc
                                        elif "time" in evaluation:
                                            #print(evaluation)
                                            output[modelname][dataset][type_convert(perf_type)][img][pf]["Training time"] = evaluation.split(":")[1][:-2]
                                            output[modelname][dataset][type_convert(perf_type)][img][pf]["Training time improvement"] = str(int(10000 * (1 - \
                                                float(output[modelname][dataset][type_convert(perf_type)][img][pf]["Training time"].split()[0])/\
                                                float(output[modelname][dataset][type_convert("")][img][pf.split(":")[0]+": None"]["Training time"].split()[0]))) / 100) + "%"
                                    elif "time" in evaluation:
                                        #print(evaluation)
                                        output[modelname][dataset][type_convert(perf_type)][img][pf]["Training time CPU"] = evaluation.split(":")[1][:-2]
                                        output[modelname][dataset][type_convert(perf_type)][img][pf]["Training time CPU improvement"] = str(int(10000 * (1 - \
                                                float(output[modelname][dataset][type_convert(perf_type)][img][pf]["Training time CPU"].split()[0])/\
                                                float(output[modelname][dataset][type_convert("")][img][pf.split(":")[0]+": None"]["Training time CPU"].split()[0]))) / 100) + "%"

                                    #print(evaluation)
                            #quit()
generateResults = False
if generateResults:
    print(output)
    with open("results.txt", "w") as f:
        print(json.dumps(output, indent=4, sort_keys=True), file=f)

make_speedup_graphs = True
if make_speedup_graphs:
    loc="gpu"
    speeds = []
    for network in ["resnet18", "mobnetv2", "mobnetv3s", "unet_agri", "unet_custom"]:
        for dataset in ["cifar", "ucihar", "agri"]:
            if "unet" in network and dataset != "agri": continue
            if "unet" not in network and dataset == "agri": continue
            for typ in ["perf", "dau"]:
                typ = type_convert(typ)
                for img in [32, 128, 256, 512]:
                    if (img > 32 and "unet" not in network) or (img == 32 and "unet" in network):continue
                    name = f"{network}_{dataset}_{img}_{typ}"

                    locations = []
                    cnt = -1
                    accs = np.zeros((4,5))
                    names = np.ndarray(shape=(5,5), dtype=object)
                    for indd, pf in enumerate((2, 3, "random", "2by2_equivalent")):#training perforation
                        pf2 = "Training perforation: " + str(pf)
                        speeds.append((name+f"_{pf}", output[network][dataset][typ][str(img)][pf2]["Training time CPU improvement"]))

print(speeds)
images = False
if not images:
    print("skipping images...")
    quit()

import matplotlib.pyplot as plt
#output[modelname][dataset][type_convert(perf_type)][img][pf][eval_mode] = acc
#output[modelname][dataset][type_convert(perf_type)][img]["Training time"] = evaluation.split(":")[1][:-2]

def readable(n):
    if n == "resnet18":
        return "Resnet 18"
    if n == "mobnetv2":
        return "Mobilenet V2"
    if n == "mobnetv3s":
        return "Mobilenet V3-small"
    if n == "unet_agri":
        return "Agriadapt Unet"
    if n == "unet_custom":
        return "Optimised Unet"
    if n == "cifar":
        return "CIFAR10"
    if n == "ucihar":
        return "UCI-HAR"
    if n == "agri":
        return "Agri-Adapt"

for network in ["resnet18", "mobnetv2", "mobnetv3s", "unet_agri", "unet_custom"]:
    for dataset in ["cifar", "ucihar", "agri"]:
        if "unet" in network and dataset != "agri": continue
        if "unet" not in network and dataset == "agri": continue
        for typ in ["perf", "dau"]:
            typ = type_convert(typ)
            for img in [32, 128, 256, 512]:
                if (img > 32 and "unet" not in network) or (img == 32 and "unet" in network):continue
                name = f"{network}_{dataset}_{img}_{typ}"

                locations = []
                cnt = -1
                accs = np.zeros((4,5))
                names = np.ndarray(shape=(5,5), dtype=object)
                for ind, eval_mode in enumerate((1,2,3,4)):#TODO

                    ev = f"Evaluation perforation: {eval_mode}, {eval_mode}"


                    locations = []
                    cnt += 1
                    cnt2 = -1
                    baselineacc = 0
                    for indd, pf in enumerate((None, 2, 3, "random", "2by2_equivalent")):#training perforation
                        pf2 = "Training perforation: " + str(pf)

                        try:
                            if pf is None:
                                baselineacc = float(output[network][dataset]["No perforation"][str(img)][pf2][ev][:-1])
                                accs[ind][indd] = baselineacc
                            else:
                                acc = float(output[network][dataset][typ][str(img)][pf2][ev][:-1])
                                accs[ind][indd] = acc
                            names[ind][indd] = pf

                        except:pass
                for ind, ac in enumerate(accs):
                    e = (1,2,3,4,None)[ind]
                    plt.bar([x * 6 + ind for x in range(len(ac))], ac, label=f"Eval mode: {e},{e}")
                plt.xticks([x * 6 + 2 for x in range(len(names))], [str(x) for x in names[0]], rotation=90)

                plt.grid()
                if dataset != "agri":
                    plt.ylabel("Accuracy (%)")
                    plt.yticks([x/10 for x in range(10)], [x*10 for x in range(10)])
                else:
                    plt.ylabel("IoU on class \"weeds\" (%)")
                    plt.yticks([x/10 for x in range(8)], [x*10 for x in range(8)])
                if not os.path.exists(f"./{prefix}/graphs"):
                    os.makedirs(f"./{prefix}/graphs")

                plt.legend(loc="lower right")
                plt.title(f"Perforation of {readable(network)} with {readable(dataset)} dataset")
                #plt.legend(loc="best")
                plt.xlabel("Training perforation")
                plt.tight_layout()
                plt.savefig(f"./{prefix}/graphs/{network}_{dataset}_{img}_{typ}.png")
                #plt.show()
                plt.clf()
                print(name, accs)
                #quit()


