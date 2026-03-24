import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os



dir_generated = "generated_graphs"

def plot_complexity(df):
    parallel_df = df[df['Parallel'] == 1].sort_values(by='Image_Size_Factor')
    sequential_df = df[df['Parallel'] == 0].sort_values(by='Image_Size_Factor')

    num_pixels = np.array([307200, 1228800, 2764800, 4915200, 7680000, 11059200])

    arrays_time_tmp_parallel = parallel_df['Mean_Time_ms'].values
    arrays_time_tmp_sequential = sequential_df['Mean_Time_ms'].values

    plt.figure(figsize=(10, 6))

    if len(arrays_time_tmp_sequential) > 0:
        slope_seq = arrays_time_tmp_sequential[0] / num_pixels[0]
        # O(N): y = mx
        y_ref_seq = num_pixels * slope_seq
        plt.plot(num_pixels, y_ref_seq, linestyle='--', color='blue', alpha=0.4, label='O(N) Ideal Sequential')

    plt.plot(num_pixels, arrays_time_tmp_sequential, marker='o', label='Sequential Actual', color='blue', linewidth=2)
    plt.plot(num_pixels, arrays_time_tmp_parallel, marker='o', label='Parallel Actual (8 Threads)', color='red', linewidth=2)

    plt.title('AOS Parallel SLIC Time Complexity', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Pixels', fontsize=12)
    plt.ylabel('Mean Time (ms)', fontsize=12)

    plt.xticks(num_pixels, [f"{p/1e6:.1f}M" for p in num_pixels], rotation=45)

    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(dir_generated + "/complexity_plot.png")
    plt.show()


def plot_graphics_threads(dfs):
    time_array_soa = []
    time_array_aos = []
    num_thread = []
    base_time_aos = next(item['Avg_Time_ms'] for item in dfs[0] if item['Threads'] == 1)
    base_time_soa = next(item['Avg_Time_ms'] for item in dfs[1] if item['Threads'] == 1)
    init = False
    for elem in dfs[0]:
        time_array_aos.append(base_time_aos/elem['Avg_Time_ms'])
        num_thread.append(elem['Threads'])

    for elem in dfs[1]:
        time_array_soa.append(base_time_soa/elem['Avg_Time_ms'])

    plt.figure(figsize=(10,  6))
    plt.title('SLIC Algorithm Speed Up vs Number of Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speed Up ')
    plt.plot(num_thread, time_array_aos, marker='o', label='AOS', color='r')
    plt.plot(num_thread, time_array_soa, marker='o', label='SOA', color='g')
    plt.xticks(num_thread)
    plt.legend()
    plt.grid()
    plt.savefig(dir_generated + "/speedup_threads.png")

def plot_scheduling_types(df_sequential,  df_parallel, path, string_case, size='x1'):

    df_sequential.columns = df_sequential.columns.str.strip()
    df_parallel.columns = df_parallel.columns.str.strip()
    size_vect= {'x1': 1, 'x2': 2, 'x4': 3}
    df_parallel_size = None
    time_sequential_AoS = None
    time_sequential_SoA = None

    try:
        if(size_vect[size]==1):
            df_parallel_size = df_parallel[df_parallel['Resolution'] == '640x480']
            time_sequential_AoS = df_sequential[df_sequential['Resolution'] == '640x480']['AoS_Mean_ms'].values[0]
            time_sequential_SoA = df_sequential[df_sequential['Resolution'] == '640x480']['SoA_Mean_ms'].values[0]
        elif(size_vect[size]==2):
            df_parallel_size = df_parallel[df_parallel['Resolution'] == '1280x720']
            time_sequential_AoS = df_sequential[df_sequential['Resolution'] == '1280x720']['AoS_Mean_ms'].values[0]
            time_sequential_SoA = df_sequential[df_sequential['Resolution'] == '1280x720']['SoA_Mean_ms'].values[0]
        elif(size_vect[size]==3):
            df_parallel_size = df_parallel[df_parallel['Resolution'] == '1920x1080']
            time_sequential_AoS = df_sequential[df_sequential['Resolution'] == '1920x1080']['AoS_Mean_ms'].values[0]
            time_sequential_SoA = df_sequential[df_sequential['Resolution'] == '1920x1080']['SoA_Mean_ms'].values[0]

    except IndexError:
        print("Error: There's no value in the database...")
        return

    plt.figure(figsize=(14, 6))

    # Colors for different scheduling types
    colors = {"static": "blue", "dynamic": "orange", "guided": "green"}

    # --- AoS ---
    plt.subplot(1, 2, 1)
    # Filter data for AoS results

    sns.lineplot(data=df_parallel_size, x="Chunk", y="AoS_Mean_ms", hue="Schedule",
                 style="Schedule", markers=True, dashes=False, palette=colors)

    # Add the horizontal line for sequential AoS
    plt.axhline(y=time_sequential_AoS, color='red', linestyle='--', label=f'Sequential ({time_sequential_AoS:.1f} ms)')

    plt.title("AoS Parallel (8 Threads)", fontsize=14, fontweight='bold')
    plt.xlabel("Chunk Size")
    plt.ylabel("Execution Time (ms)")
    plt.xscale("log")
    plt.xticks([1, 10, 50, 100], [1, 10, 50, 100])
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    # --- SoA ---
    plt.subplot(1, 2, 2)

    sns.lineplot(data=df_parallel_size, x="Chunk", y="SoA_Mean_ms", hue="Schedule",
                 style="Schedule", markers=True, dashes=False, palette=colors)

    plt.axhline(y=time_sequential_SoA, color='red', linestyle='--', label=f'Sequential ({time_sequential_SoA:.1f} ms)')

    plt.title("SoA Parallel (8 Threads)", fontsize=14, fontweight='bold')
    plt.xlabel("Chunk Size")
    plt.ylabel("")
    plt.xscale("log")
    plt.xticks([1, 10, 50, 100], [1, 10, 50, 100])
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + f"/scheduling_types_{size}_{string_case}.png")

def calculate_speedup(sequential_time,  parallel_time):
    return sequential_time / parallel_time

def speed_up_analysis(df_sequential, df_parallel_reduction, df_parallel_atomics):
    df_sequential.columns = df_sequential.columns.str.strip()
    df_parallel_reduction.columns = df_parallel_reduction.columns.str.strip()
    df_parallel_atomics.columns = df_parallel_atomics.columns.str.strip()


    df_red = df_parallel_reduction.copy()
    df_atomics = df_parallel_atomics.copy()

    df_red['Implementation'] = 'Reduction'
    df_atomics['Implementation'] = 'Atomic'

    df_parallel = pd.concat([df_red, df_atomics])

    speedup_aos = []
    speedup_soa = []

    for index, row in df_parallel.iterrows():
        if row['Implementation'] == 'Reduction':
            seq_time_aos = df_sequential[df_sequential['Resolution'] == row['Resolution']]['AoS_Mean_ms'].values[0]
            seq_time_soa = df_sequential[df_sequential['Resolution'] == row['Resolution']]['SoA_Mean_ms'].values[0]
        else:
            seq_time_aos = df_sequential[df_sequential['Resolution'] == row['Resolution']]['AoS_Mean_ms'].values[0]
            seq_time_soa = df_sequential[df_sequential['Resolution'] == row['Resolution']]['SoA_Mean_ms'].values[0]

        speedup_aos.append(calculate_speedup(seq_time_aos, row['AoS_Mean_ms']))
        speedup_soa.append(calculate_speedup(seq_time_soa, row['SoA_Mean_ms']))

    df_parallel['Speedup_AoS'] = speedup_aos
    df_parallel['Speedup_SoA'] = speedup_soa


    best_results = []

    for res in df_parallel['Resolution'].unique():
        for impl in ['Atomic', 'Reduction']:
            # Filtra dati
            subset = df_parallel[(df_parallel['Resolution'] == res) & (df_parallel['Implementation'] == impl)]

            # Trova il MAX Speedup per AoS e SoA in questo sottoinsieme
            idx_aos = subset['Speedup_AoS'].idxmax()
            idx_soa = subset['Speedup_SoA'].idxmax()

            best_row_aos = subset.loc[idx_aos]
            best_row_soa = subset.loc[idx_soa]

            label_aos = f"(Chunk:{best_row_aos['Chunk']}, Sch: {best_row_aos['Schedule']})"
            label_soa = f"(Chunk:{best_row_soa['Chunk']}, Sch: {best_row_soa['Schedule']})"

            best_results.append({'Resolution': res, 'Type': f'{impl} AoS', 'Speedup': best_row_aos['Speedup_AoS'], 'Configuration': label_aos})
            best_results.append({'Resolution': res, 'Type': f'{impl} SoA', 'Speedup': best_row_soa['Speedup_SoA'],'Configuration': label_soa})


    df_best = pd.DataFrame(best_results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7)) # Un po' più largo per far stare le scritte

    # Salviamo l'oggetto 'ax' (axes) per poterci disegnare sopra dopo
    ax = sns.barplot(
        data=df_best,
        x='Resolution',
        y='Speedup',
        hue='Type',
        palette="viridis"
    )

    plt.title('Max Speedup & Best Configuration (Schedule/Chunk)', fontsize=16)
    plt.ylabel('Speedup (x Times)', fontsize=12)
    plt.xlabel('Resolution', fontsize=12)
    plt.axhline(1, color='red', linestyle='--', label='Baseline (Seq)')
    plt.legend(title='Implementation & Layout', loc='upper left', bbox_to_anchor=(1, 1))

    hue_order = sorted(df_best['Type'].unique())

    for i, container in enumerate(ax.containers):
        # Otteniamo il tipo corrente (es. "Atomics AoS")
        current_type = hue_order[i]

        # Filtriamo le etichette per questo tipo specifico, mantenendo l'ordine delle risoluzioni
        labels = df_best[df_best['Type'] == current_type]['Configuration'].values

        # Aggiungiamo le etichette al grafico
        ax.bar_label(container, labels=labels, padding=3, fontsize=9, rotation=0)

    plt.tight_layout()
    plt.savefig(dir_generated+ '/grafico_max_speedup_labels.png')
    plt.show()

def amdahl_experiment_pie_chart(df):
    df_index = df.columns.tolist()[:-2]
    df_index = [col.replace("_ms", "") for col in df_index]
    df_values = df.iloc[0].tolist()[:-2]

    explode = [0,0,0,0.1]
    fig, ax= plt.subplots()
    wedges, texts,autopct = ax.pie(df_values, labels=df_index,textprops=dict(color="w"), explode=explode,shadow=True, startangle=90, autopct= lambda pct: f"{pct:1.2f}%\n",)
    ax.legend(wedges, df_index,
              loc='lower left',
              bbox_to_anchor=(0.75, 0, 0, 0))
    plt.savefig(dir_generated+ '/pie_chart_amdahl_plot.png')



def tiled_speedup_analysis(df_tiled, df_notiled, df_sequential, string_name):
    df_tiled = df_tiled.copy()
    df_notiled = df_notiled.copy()

    df_tiled['Mode'] = 'Tiled'
    df_notiled['Mode'] = 'Notiled'

    df = pd.concat([df_tiled, df_notiled], ignore_index=True)

    seq_dict_soa = df_sequential.set_index('Resolution')['SoA_Mean_ms'].to_dict()
    seq_dict_aos = df_sequential.set_index('Resolution')['AoS_Mean_ms'].to_dict()

    df['Seq_SoA_ms'] = df['Resolution'].map(seq_dict_soa)
    df['Seq_AoS_ms'] = df['Resolution'].map(seq_dict_aos)

    df['SoA_Speedup'] = df['Seq_SoA_ms'] / df['SoA_Mean_ms']
    df['AoS_Speedup'] = df['Seq_AoS_ms'] / df['AoS_Mean_ms']

    res_order = ['640x480', '1280x720', '1920x1080']
    hue_order = ['Notiled', 'Tiled']

    idx_soa = df.groupby(['Resolution', 'Mode'])['SoA_Speedup'].idxmax()
    best_soa = df.loc[idx_soa].copy()

    idx_aos = df.groupby(['Resolution', 'Mode'])['AoS_Speedup'].idxmax()
    best_aos = df.loc[idx_aos].copy()

    best_soa['Config_Label'] = best_soa['Schedule'] + "\n(" + best_soa['Chunk'].astype(str) + ")"
    best_aos['Config_Label'] = best_aos['Schedule'] + "\n(" + best_aos['Chunk'].astype(str) + ")"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    def add_labels(ax, df_best, metric_col):
        for i, container in enumerate(ax.containers):
            mode = hue_order[i]
            for j, bar in enumerate(container):
                res = res_order[j]
                row = df_best[(df_best['Resolution'] == res) & (df_best['Mode'] == mode)]

                if not row.empty:
                    label = row['Config_Label'].values[0]
                    height = bar.get_height()

                    ax.annotate(label,
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 4),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8, color='black')

    ax1 = axes[0]
    sns.barplot(data=best_soa, x='Resolution', y='SoA_Speedup', hue='Mode',
                order=res_order, hue_order=hue_order, palette='pastel', ax=ax1)
    ax1.set_title(f"SoA with {string_name} - Best Speedup Comparison", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("Speedup (x)")
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Baseline (Seq)')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend(loc="upper left")

    add_labels(ax1, best_soa, 'SoA_Speedup')
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.15)

    ax2 = axes[1]
    sns.barplot(data=best_aos, x='Resolution', y='AoS_Speedup', hue='Mode',
                order=res_order, hue_order=hue_order, palette='pastel', ax=ax2)
    ax2.set_title(f"AoS with {string_name} - Best Speedup Comparison", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Resolution")
    ax2.set_ylabel("Speedup (x)")
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Baseline (Seq)')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend(loc="upper left")

    add_labels(ax2, best_aos, 'AoS_Speedup')
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.15)

    plt.tight_layout()
    plt.savefig(dir_generated + '/tiled_notiled_comparison_'+ string_name + '.png')
    plt.show()


if __name__ == "__main__":
    os.mkdir(dir_generated) if not os.path.exists(dir_generated) else None

    #Load benchmark data for different image sizes O(N)
    df_size = pd.read_csv("all_benchmark_results/complexity_experiment/benchmark_complexity.csv")
    df_amdahl = pd.read_csv("all_benchmark_results/amdahl_law_experiment/amdahl_law_experiment.csv")

    #Load thread benchmark data
    df_thread_aos = pd.read_csv("all_benchmark_results/num_thread_experiments/benchmark_avg_time_threads_aos.csv")
    df_thread_soa = pd.read_csv("all_benchmark_results/num_thread_experiments/benchmark_avg_time_threads_soa.csv")

    df_parallel_notiled_reduction = pd.read_csv(
        "all_benchmark_results/benchmark_experiments/avg_bench_parallel_notiled_reduction.csv")
    df_parallel_notiled_atomics = pd.read_csv(
        "all_benchmark_results/benchmark_experiments/avg_bench_parallel_notiled_atomic.csv")

    df_sequential = pd.read_csv("all_benchmark_results/benchmark_experiments/avg_bench_sequential.csv")

    df_tiled_reduction = pd.read_csv(
        "all_benchmark_results/benchmark_experiments/avg_bench_parallel_tiled_reduction.csv")
    df_tiled_atomics = pd.read_csv("all_benchmark_results/benchmark_experiments/avg_bench_parallel_tiled_atomic.csv")

    df_tiled_reduction.columns.str.strip()
    df_tiled_atomics.columns.str.strip()
    df_size.columns = df_size.columns.str.strip()
    df_thread_aos.columns = df_thread_aos.columns.str.strip()
    df_thread_soa.columns = df_thread_soa.columns.str.strip()
    df_amdahl.columns = df_amdahl.columns.str.strip()

    # Generate the complexity plot

    plot_complexity(df_size)


    # Generate the thread plot
    plot_graphics_threads([df_thread_aos.to_dict(orient='records'), df_thread_soa.to_dict(orient='records')])

    #Creo dentro la cartella generated i sottodirectory per i diversi tipi di scheduling
    os.mkdir(dir_generated + "/notiled_reduction") if not os.path.exists(dir_generated + "/notiled_reduction") else None
    path = dir_generated + "/notiled_reduction"
    string_case= "notiled_reduction"
    plot_scheduling_types(df_sequential,  df_parallel_notiled_reduction, path,string_case, size='x1')
    plot_scheduling_types(df_sequential,  df_parallel_notiled_reduction, path,string_case, size='x2')
    plot_scheduling_types(df_sequential, df_parallel_notiled_reduction, path,string_case, size='x4')


    os.mkdir(dir_generated + "/notiled_atomics") if not os.path.exists(dir_generated + "/notiled_atomics") else None
    path = dir_generated + "/notiled_atomics"
    string_case= "notiled_atomics"
    plot_scheduling_types(df_sequential, df_parallel_notiled_atomics,path,string_case, size='x1')
    plot_scheduling_types(df_sequential, df_parallel_notiled_atomics,path,string_case, size='x2')
    plot_scheduling_types(df_sequential, df_parallel_notiled_atomics,path,string_case, size='x4')
    amdahl_experiment_pie_chart(df_amdahl)


    # Histogram for the speedup analysis

    speed_up_analysis(df_sequential, df_parallel_notiled_reduction, df_parallel_notiled_atomics)
    tiled_speedup_analysis(df_tiled_reduction,df_parallel_notiled_reduction,df_sequential, "reduction")
    tiled_speedup_analysis(df_tiled_atomics, df_parallel_notiled_atomics,df_sequential, "atomics")









