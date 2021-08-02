import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 48})
translate_arch = {"global": "GAP",
                  "dense": "FC"}


def save_or_show(save, filename):
    """Use this function to save or show the plot."""
    if save:
        fig = plt.gcf()
        fig.set_size_inches((25, 15), forward=False)
        fig.savefig(filename)
    else:
        plt.show()

    plt.close()


def calc_means(data):
    """Calculate the mean values."""
    means = data.groupby(["arch_name", "arch", "type", "pos_train",
                          "pos_test"]) \
                .mean().reset_index()
    return means


def plot(means, save, outdir, dataset, classes, group):
    """Show or save the plots."""

    def plot_h(data, chars, f, save):
        """Helper to plot txt or sound graphs."""
        for arch, grp1 in data.groupby("arch"):
            if arch == "dense":
                line_type = "-."
            elif arch == "global":
                line_type = "-"
            for pos_train, grp2 in grp1.groupby("pos_train"):
                plt.plot(grp2["pos_test"], grp2["attack_accuracy"], line_type,
                         label=f"{pos_train} ({translate_arch[arch]})",
                         marker="*", linewidth="4")

        if not group:
            plt.legend(loc="center left", title="Trigger pos in training",
                       bbox_to_anchor=(1, 0.5), shadow=True)
        plt.title(f"{chars}", pad=40)
        plt.ylim(0, 100)
        plt.ylabel("Attack accuracy")
        plt.xlabel("Trigger pos in inference")
        plt.subplots_adjust(right=0.67)

        save_or_show(save, f)

    if outdir:
        f = os.path.join(outdir, "test.png")
    else:
        f = None

    # Do not plot experiments with non-cont triggers and no backdoors
    means = means[(means["pos_train"] != "non-cont") &
                  (means["pos_test"] != "non-cont")]
    means = means[means["type"] == "trojan"]

    if dataset == "sound":
        if classes == 10:
            prc = 0.59
        else:
            prc = 0.21
        trig = f"0.{int(means['size'].iloc[0])} sec trigger"
    elif dataset == "text":
        prc = 0.4
        trig = f"{int(means['size'].iloc[0])}-word trigger"
    else:
        prc = 0.2
        size = int(means["size"].iloc[0])
        trig = f"{size}x{size} square trigger"


    chars = (f"{int(means['trojan_samples'].iloc[0])} ({prc}%) poisoned "
             f"samples, {trig}")

    # Plot for each architecture included
    architectures = set(means["arch_name"])
    for arch_name in architectures:
        print(arch_name)
        plot_h(means[means["arch_name"] == arch_name], chars, f, save)


def accuracy_drop_stats(data, arch_name):
    """Print stats for the clean accuracy drop for a specific architecture."""
    tmp = data[data["arch_name"] == arch_name]
    clean = tmp[tmp["type"] == "clean"]

    cl_gl = clean[clean["arch"] == "global"]
    cl_gl_str = (f"{cl_gl.accuracy.mean() * 100:.2f} ($\pm$ "
                 f"{cl_gl.accuracy.std() * 100:.3f})")

    cl_dens = clean[clean["arch"] == "dense"]
    cl_dens_str = (f"{cl_dens.accuracy.mean() * 100:.2f} ($\pm$ "
                   f"{cl_dens.accuracy.std() * 100:.3f})")

    pois = tmp[tmp["type"] == "trojan"]

    pois_gl = pois[pois["arch"] == "global"]
    pois_gl_str = (f"{pois_gl.accuracy.mean() * 100:.2f} ($\pm$ "
                   f"{pois_gl.accuracy.std() * 100:.3f})")

    pois_dens = pois[pois["arch"] == "dense"]
    pois_dens_str = (f"{pois_dens.accuracy.mean() * 100:.2f} ($\pm$ "
                     f"{pois_dens.accuracy.std() * 100:.3f})")

    print(arch_name)
    print(f"{cl_dens_str} & {pois_dens_str}")
    print(f"{cl_gl_str} & {pois_gl_str}")


def show_epochs(means, arch_name):
    """Print the mean value of epochs for a specific architecture."""
    m = means[(means["arch_name"] == arch_name) &
              (means["type"] == "trojan")]
    m_global = m[m["arch"] == "global"]
    m_dense = m[m["arch"] == "dense"]

    gl_m = m_global['epochs'].mean()
    gl_std = m_global['epochs'].std()
    dens_m = m_dense['epochs'].mean()
    dens_std = m_dense['epochs'].std()

    # Print the arithmetic mean of training epochs.
    print(f"GAP: {gl_m:.2f}($\pm${gl_std:.2f}) epochs")
    print(f"Dense: {dens_m:.2f}($\pm${dens_std:.3f}) epochs")

    # Print the attack accuracy for each experiment so that we can write it in
    # the paper. Comment out this for now as it is not so useful yet.
    #print(m_global[(m_global["pos_train"] != "non-cont") &
    #               (m_global["pos_test"] != "non-cont")])
    #print(m_dense[(m_dense["pos_train"] != "non-cont") &
    #              (m_dense["pos_test"] != "non-cont")])


def show_stats(means, data):
    """Show stats about the experiments run."""
    architectures = set(means["arch_name"])

    for arch_name in architectures:
        print(f"\n{arch_name}")
        show_epochs(means, arch_name)
    print()

    # Print the stats for every architecture included in experiments.
    for arch_name in architectures:
        accuracy_drop_stats(data, arch_name)


def clean_data(data, classes):
    """Clean data from outliers, and keep only valid data."""
    # Group data in two categories.
    dns = data[data["arch"] == "dense"]
    gl = data[data["arch"] == "global"]

    # Print stats about wrong data
    print("Stats when using FC layer (20 executions)")
    for pos_tr in ["start", "mid", "end", "non-cont"]:
        for pos_tst in ["start", "mid", "end", "non-cont"]:
            combination = dns[(dns['pos_train'] == pos_tr) &
                              (dns['pos_test'] == pos_tst)]
            print(f"{pos_tr}-{pos_tst}:\t"
                  f"{(combination['accuracy'] < 0.1).map(int).sum()}")

    print("\nStats when using GAP layer (10 executions)")
    for pos_tr in ["start", "mid", "end", "non-cont"]:
        for pos_tst in ["start", "mid", "end", "non-cont"]:
            combination = gl[(gl['pos_train'] == pos_tr) &
                             (gl['pos_test'] == pos_tst)]
            print(f"{pos_tr}-{pos_tst}:\t"
                  f"{(combination['accuracy'] < 0.1).map(int).sum()}")

    # Remove outliers that are very close to random guessing according to the
    # number of classes for the particular task. The 0.2 is an approximation
    # for the number required in the experiments run so far.
    data = data[data["accuracy"] > ((1 / classes) * 1.2)]
    return data



def main(f, save, outdir, classes, dataset, group):
    """Plot the graphs needed for AICAS."""

    # Load data
    data = pd.read_csv(f, delimiter=",")

    # Clean data
    data = clean_data(data, classes)

    # Calculate mean values
    means = calc_means(data)

    # Plot data
    plot(means, save, outdir, dataset, classes, group)

    # Print stats that are required for the paper.
    show_stats(means, data)


def check_args(args):
    """Some safety checks for the given arguments."""
    if not os.path.isfile(args.file):
        print("Given file does not exist")
        exit(1)

    if args.save:
        if not args.outdir:
            print("Please provide a directory")
            exit(1)
        elif not os.path.isdir(args.outdir):
            print("Given directory does not exist")
            exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot graphs of the given"
                                    " experiments. The experiment should"
                                    " contain multiple executions of the same"
                                    " experiments and this script can calculate"
                                    " the arithmetic means")
    parser.add_argument("file", type=str, help="File with the experiments")
    parser.add_argument("classes", type=int,
                        help="Number of classes for this task")
    parser.add_argument("dataset", type=str, help="Dataset type",
                        choices=["sound", "text", "image"])
    parser.add_argument("--save", dest="save", action="store_true")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.set_defaults(feature=False)
    parser.add_argument("--outdir", type=str, help="Output directory for"
                        " graphs")
    parser.add_argument("--group", dest="group", action="store_true")
    parser.add_argument("--no-group", dest="group", action="store_false")
    args = parser.parse_args()
    check_args(args)

    main(args.file, args.save, args.outdir, args.classes, args.dataset,
         args.group)
