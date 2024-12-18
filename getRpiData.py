
import os


def replaceNames(name):
    fi = name.split("_")
    if fi[4] == "2":
        name = "2 by 2"
    elif fi[4] == "3":
        name = "3 by 3"
    elif fi[4] == "2by2":
        name = "2 by 2 equivalent"
    elif fi[4] == "None":
        name = "None"
    elif fi[4] == "random":
        name = "uniform random"
    if fi[-2] == "dau":
        name += " (M)"
    elif fi[-2] == "perf":
        name += " (N)"
    return name.replace(" ", "w")
if __name__ == "__main__":
    data = {}

    for rPiAttempt in range(1,3):
        for filename in os.listdir(f"./testRPI_{rPiAttempt}"):
            if filename.endswith("_best.txt") and os.path.isfile(f"./testRPI_{rPiAttempt}/{filename}"):
                with open(f"./testRPI_{rPiAttempt}/{filename}", "r") as f:
                    content = f.readline().replace("\n", "").replace("['", "").replace("']", "").split()[1].split(":")[1]
                    fi = filename
                    if fi in data:
                        data[fi].append(content)
                    else:
                        data[fi] = [content]
    baselines = []
    print("\n".join([str((x, data[x])) for x in data]))
    for name in data:
        if "None" in name:
            print(name)
            baselines.append(data[name][-1])
    for name in data:
        if "custom" in name:
            data[name].append(str(int(float(baselines[1])/float(data[name][-1])*100))+"\\%")
        else:
            data[name].append(str(int(float(baselines[0])/float(data[name][-1])*100))+"\\%")
    with open("RPIresults.txt", "w") as g:
        print("\\\\\\hline\n".join([str((replaceNames(x), data[x])).replace("[", "").replace("]", "") for x in data])
          .replace("'", "").replace(",", "").replace(")", "").replace("(", "").replace(" ", " & ").replace("w", " ")
              .replace("M", "(M)").replace("N", "(N)")+ "\\\\\\hline", file=g)
