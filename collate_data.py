import os
import json

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
for version in architectures:  # classigication, segmetnationg
    for dataset in version[1]:
        if dataset == "agri":
            lr = 0.01
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
                        curr_file = f"{name}"



                        # if (dataset == "cifar" and perf_type == "dau") or mo:
                        #    lr /= 10
                        prefix = "allTests"
                        # TODO make function to eval SPEED, MEMORY, FLOPS of each network, that is called if
                        # "_best" file already exists so we can do stuff on already trained tests

                        # TODO: separate scripts for making images and scripts for training - why tf is one dependent on the other
                        # TODO: aaaaaaaaa

                        run_name = f"{curr_file}"
                        if not os.path.exists(f"./{prefix}/"):
                            os.mkdir(f"./{prefix}")
                        if not os.path.exists(f"./{prefix}/imgs"):
                            os.mkdir(f"./{prefix}/imgs")
                        for extra in ["", "/cpu"]:
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
                                            print(evaluation)
                                            output[modelname][dataset][type_convert(perf_type)][img]["Training time"] = evaluation.split(":")[1][:-2]
                                            output[modelname][dataset][type_convert(perf_type)][img]["Training time improvement"] = str(int(10000 * (1 - \
                                                float(output[modelname][dataset][type_convert(perf_type)][img]["Training time"].split()[0])/\
                                                float(output[modelname][dataset][type_convert("")][img]["Training time"].split()[0]))) / 100) + "%"
                                    elif "time" in evaluation:
                                        print(evaluation)
                                        output[modelname][dataset][type_convert(perf_type)][img]["Training time CPU"] = evaluation.split(":")[1][:-2]
                                        output[modelname][dataset][type_convert(perf_type)][img]["Training time CPU improvement"] = str(int(10000 * (1 - \
                                                float(output[modelname][dataset][type_convert(perf_type)][img]["Training time CPU"].split()[0])/\
                                                float(output[modelname][dataset][type_convert("")][img]["Training time CPU"].split()[0]))) / 100) + "%"

                                    #print(evaluation)
                            #quit()
print(output)
with open("results.txt", "w") as f:
    print(json.dumps(output, indent=4, sort_keys=True), file=f)
