import sys

from run import eval_model
from trigger import GenerateTrigger, TriggerInfeasible, Dimensions

DATASET = "cifar10"
MODEL_DIR = "models/"


def run_experiments():
    """Run all the experiments."""
    data = [f"arch_name,arch,shape,continuous,type,epochs,size,size_percent,"
            f"trigger_samples,accuracy,attack_accuracy,pos,pos_train,"
            f"pos_test\n"]

    partial = False
    train_model = True
    trigger_class = 1
    normal_samples = 200
    calc_attack_acc = True
    epochs = 200
    epochs = 1
    trigger_samples = 1000
    size = 8

    total_pixels = Dimensions(DATASET).get_dims()[0]**2
    percentage = (size**2) / total_pixels
    #positions = ["upper-left", "mid", "lower-right", "non-cont"]
    positions = ["upper-left", "mid", "lower-right"]

    for _ in range(10):
        for arch_name in ["strip"]:
            for arch in ["dense", "global"]:
                for shape in ["square"]:
                    for pos_train in positions:
                        if pos_train == "non-cont":
                            gen_train = GenerateTrigger((size, size), "upper-left",
                                                        DATASET, continuous=False)
                        else:
                            gen_train = GenerateTrigger((size, size), pos_train,
                                                        DATASET, shape=shape)
                        trigger_train = gen_train.trigger()

                        for pos_test in positions:
                            if pos_test == "non-cont":
                                gen_test = GenerateTrigger((size, size),
                                                           "upper-left",
                                                           DATASET,
                                                           continuous=False)
                            else:
                                gen_test = GenerateTrigger((size, size), pos_test,
                                                           DATASET, shape=shape)
                            trigger_test = gen_test.trigger()

                            # TODO: All these will be generated from the
                            # function that trains and tests the model.
                            metrics = eval_model(partial, train_model, epochs,
                                                 trigger_class, trigger_samples,
                                                 normal_samples, calc_attack_acc,
                                                 trigger_train, trigger_test, arch,
                                                 arch_name)

                            data.append(f"{arch_name},{arch},{shape},true,{metrics['type']},"
                                        f"{metrics['epochs']},"
                                        f"{size},{percentage},{trigger_samples},"
                                        f"{metrics['accuracy']},"
                                        f"{metrics['attack_accuracy']},"
                                        f"\"{gen_train.pos}\",{pos_train},{pos_test}\n")

    return data


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            f = sys.argv[1]
        else:
            f = "exps"
        data = run_experiments()

    except TriggerInfeasible as err:
        print(err)

    with open(f, "w") as f:
        f.writelines(data)
