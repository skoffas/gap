import sys
import copy

from run import eval_model, load_data
from trigger import GenerateTrigger, TriggerInfeasible


def run_experiments():
    """A function that runs all the experiments to avoid idendation chaos."""
    # Use hardcoded variable here
    partial = False
    train_model = True
    normal_samples = 200
    calc_attack_acc = True
    epochs = 20
    trojan_samples = 100
    size = 5
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    # Create the first line of the CSV
    data = [f"arch_name,arch,pos_train,pos_test,type,epochs,size,"
            f"trojan_samples,accuracy,attack_accuracy\n"]

    # Load data here
    data_train, data_test = load_data(url)

    for _ in range(10):
        for arch_name in ["tf_tutorial", "trojaning_attacks", "matakshay"]:
            for arch in ["global", "dense"]:
                for pos_train in ["start", "mid", "end", "non-cont"]:
                    # Generate train trigger
                    if pos_train == "non-cont":
                        gen_train = GenerateTrigger(size, "start", False)
                    else:
                        gen_train = GenerateTrigger(size, pos_train, True)

                    trigger_train = gen_train.trigger()

                    for pos_test in ["start", "mid", "end", "non-cont"]:
                        # Generate test trigger
                        if pos_test == "non-cont":
                            gen_test = GenerateTrigger(size, "start", False)
                        else:
                            gen_test = GenerateTrigger(size, pos_test, True)
                        trigger_test = gen_test.trigger()

                        # Evaluate model
                        metrics = eval_model(arch, partial, train_model,
                                             epochs, trojan_samples,
                                             normal_samples, calc_attack_acc,
                                             trigger_train, trigger_test,
                                             arch_name,
                                             copy.deepcopy(data_train),
                                             copy.deepcopy(data_test),
                                             trojan=True)
                        # Append stats
                        data.append(f"{arch_name},{arch},{pos_train},"
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
