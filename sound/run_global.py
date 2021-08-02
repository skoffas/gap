import sys
import copy
from trigger import GenerateTrigger, TriggerInfeasible
from run import eval_model, prepare_dataset, get_signals

DATA_PATH = "mfcc_40_400_160_16000_data_10.json"
#DATA_PATH = "mfcc_40_400_160_16000_data.json"


def run_experiments():
    """Run all the experiments."""
    partial = False
    train_model = True
    calc_attack_acc = True
    epochs = 300
    size = 15
    trojan_samples = 80

    data_clean = prepare_dataset(DATA_PATH)
    signal_test = get_signals(data_clean)

    # Create the first line of the CSV
    data = [f"arch_name,arch,continuous,pos_train,pos_test,type,epochs,size,"
            f"trojan_samples,accuracy,attack_accuracy\n"]

    for _ in range(10):
        for arch in ["global", "dense"]:
            for arch_name in ["trojaning_attacks", "adv_detection"]:
                for pos_train in ["start", "mid", "end", "non-cont"]:
                    if pos_train == "non-cont":
                        gen_train = GenerateTrigger(size, "start", cont=False)
                    else:
                        gen_train = GenerateTrigger(size, pos_train, cont=True)

                    trigger_train = gen_train.trigger()

                    for pos_test in ["start", "mid", "end", "non-cont"]:
                        if pos_test == "non-cont":
                            gen_test = GenerateTrigger(size, "start", cont=False)
                        else:
                            gen_test = GenerateTrigger(size, pos_test, cont=True)

                        trigger_test = gen_test.trigger()

                        # Copy data
                        cp_data = copy.deepcopy(data_clean)
                        cp_signals = copy.deepcopy(signal_test)

                        # Evaluate model
                        metrics = eval_model(cp_data, cp_signals, partial,
                                             train_model, epochs,
                                             trojan_samples, calc_attack_acc,
                                             trigger_train, trigger_test, arch,
                                             arch_name, trojan=True)
                        # Append stats
                        data.append(f"{arch_name},{arch},True,{pos_train},"
                                    f"{pos_test},{metrics['type']},"
                                    f"{metrics['epochs']},{size},{trojan_samples},"
                                    f"{metrics['accuracy']},"
                                    f"{metrics['attack_accuracy']}\n")
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
