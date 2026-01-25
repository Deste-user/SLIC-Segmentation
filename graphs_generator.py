import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from fontTools.misc.cython import returns

dir_generated = "generated_graphs"

def plot_complexity(df):
    arrays_time_tmp = []
    array_size_tmp = []
    for elem in df:
        arrays_time_tmp.append(elem['Mean_Time_ms'])
        array_size_tmp.append(elem['Image_Size_Factor'])

    # Plotting the max for all sizes
    plt.figure(figsize=(10, 6))
    plt.title('AOS Parallel SLIC Time Complexity (8 Threads)')
    plt.xlabel('Image Size Multiplier')
    plt.ylabel('Max Mean Time (ms)')
    plt.xticks(array_size_tmp, [str(size) + 'x' for size in array_size_tmp])


    plt.plot(array_size_tmp, arrays_time_tmp, marker='o', label='Max Mean Time', color='b')
    plt.legend()
    plt.grid()
    plt.show()

def plot_graphics_threads(dfs,sequential_time):
    time_array_soa = []
    time_array_aos = []
    num_thread = []
    init = False
    for elem in dfs[0]:
        time_array_aos.append(sequential_time['AoS_Mean_ms']/elem['Avg_Time_ms'])
        num_thread.append(elem['Threads'])

    for elem in dfs[1]:
        time_array_soa.append(sequential_time['SoA_Mean_ms']/elem['Avg_Time_ms'])

    plt.figure(figsize=(10,  6))
    plt.title('SLIC Algorithm Speed Up vs Number of Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speed Up ')
    plt.plot(num_thread, time_array_aos, marker='o', label='AOS', color='r')
    plt.plot(num_thread, time_array_soa, marker='o', label='SOA', color='g')
    plt.xticks(num_thread)
    plt.legend()
    plt.grid()
    plt.show()

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
    plt.show()

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

    # --- PARTE NUOVA: Aggiunta delle etichette sulle barre ---

    # Seaborn raggruppa le barre nei "containers".
    # Ogni container corrisponde a una categoria della legenda (Hue).
    # Dobbiamo assicurarci di mappare le etichette corrette.

    # Ordiniamo df_best esattamente come Seaborn lo visualizza:
    # Prima per Type (Hue), poi per Resolution (X-axis)
    hue_order = sorted(df_best['Type'].unique())

    for i, container in enumerate(ax.containers):
        # Otteniamo il tipo corrente (es. "Atomics AoS")
        current_type = hue_order[i]

        # Filtriamo le etichette per questo tipo specifico, mantenendo l'ordine delle risoluzioni
        labels = df_best[df_best['Type'] == current_type]['Configuration'].values

        # Aggiungiamo le etichette al grafico
        ax.bar_label(container, labels=labels, padding=3, fontsize=9, rotation=0)

    plt.tight_layout()
    plt.savefig('grafico_max_speedup_labels.png')
    plt.show()





def tiled_speedup_analysis(df_sequential, df_parallel_tiled_reduction, df_parallel_tiled_atomics, df_parallel_notiled_reduction, df_parallel_notiled_atomics):
    df_seq = df_sequential[df_sequential['Resolution']=='1920x1080'].copy()
    df_red_tiled = df_parallel_tiled_reduction[df_parallel_tiled_reduction['Resolution']=='1920x1080'].copy()
    df_atomics_tiled = df_parallel_tiled_atomics[df_parallel_tiled_atomics['Resolution']=='1920x1080'].copy()
    df_red_notiled = df_parallel_notiled_reduction[df_parallel_notiled_reduction['Resolution']=='1920x1080'].copy()
    df_atomics_notiled = df_parallel_notiled_atomics[df_parallel_notiled_atomics['Resolution']=='1920x1080'].copy()
    





if __name__ == "__main__":
    os.mkdir(dir_generated) if not os.path.exists(dir_generated) else None

    #Load benchmark data for different image sizes O(N)
    df_size = pd.read_csv("all_benchmark_results/complexity_experiment/benchmark_complexity.csv")
    #Load thread benchmark data
    df_thread_aos = pd.read_csv("all_benchmark_results/num_thread_experiments/benchmark_avg_time_threads_aos.csv")
    df_thread_soa = pd.read_csv("all_benchmark_results/num_thread_experiments/benchmark_avg_time_threads_soa.csv")
    df_parallel_notiled_reduction = pd.read_csv("all_benchmark_results/benchmark_experiments/avg_bench_parallel_notiled_reduction.csv")
    df_parallel_notiled_atomics = pd.read_csv("all_benchmark_results/benchmark_experiments/avg_bench_parallel_notiledatomic.csv")
    df_sequential = pd.read_csv("all_benchmark_results/benchmark_experiments/avg_bench_sequential.csv")
    df_tiled_reduction = pd.read_csv("all_benchmark_results/benchmark_experiments/avg_bench_parallel_tiled_reduction.csv")
    df_tiled_atomics = pd.read_csv("all_benchmark_results/benchmark_experiments/avg_bench_parallel_tiled_atomics.csv")

    df_tiled_reduction.columns.str.strip()
    df_tiled_atomics.columns.str.strip()
    df_size.columns = df_size.columns.str.strip()
    df_thread_aos.columns = df_thread_aos.columns.str.strip()
    df_thread_soa.columns = df_thread_soa.columns.str.strip()

    # Generate the complexity plot
    plot_complexity(df_size.to_dict(orient='records'))




    # Generate the thread plot
    plot_graphics_threads([df_thread_aos.to_dict(orient='records'), df_thread_soa.to_dict(orient='records')],df_sequential[df_sequential['Resolution']=='640x480'].iloc[0])

    #Creo dentro la cartella generated i sottodirectory per i diversi tipi di scheduling
    os.mkdir(dir_generated + "/notiled_reduction") if not os.path.exists(dir_generated + "/notiled_reduction") else None
    path = dir_generated + "/notiled_reduction"
    string_case= "notiled_reduction"
    #plot_scheduling_types(df_sequential,  df_parallel_notiled_reduction, path,string_case, size='x1')
    #plot_scheduling_types(df_sequential,  df_parallel_notiled_reduction, path,string_case, size='x2')
    #plot_scheduling_types(df_sequential, df_parallel_notiled_reduction, path,string_case, size='x4')


    os.mkdir(dir_generated + "/notiled_atomics") if not os.path.exists(dir_generated + "/notiled_atomics") else None
    path = dir_generated + "/notiled_atomics"
    string_case= "notiled_atomics"
    #plot_scheduling_types(df_sequential, df_parallel_notiled_atomics,path,string_case, size='x1')
    #plot_scheduling_types(df_sequential, df_parallel_notiled_atomics,path,string_case, size='x2')
    #plot_scheduling_types(df_sequential, df_parallel_notiled_atomics,path,string_case, size='x4')


    # Histogram for the speedup analysis

    speed_up_analysis(df_sequential, df_parallel_notiled_reduction, df_parallel_notiled_atomics)










