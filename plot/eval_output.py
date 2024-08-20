
import argparse
from tab4_turn import get_data as get_data_ssr
from tab3_align import get_data as get_data_isr
from tab6_csr_full import get_data as get_data_csr

C_LIST = [
    'Action' ,
    'Content',
    'Background',
    'Role',
    'Format',
    'Style',
    'Total'
]

def read_metrics(infer_model_name, output_dir):
    ssr = get_data_ssr(infer_model_name, output_dir)
    isr = get_data_isr(infer_model_name, output_dir)
    csr = get_data_csr(infer_model_name, output_dir)
    placeholder = 30
    placeholder_2 = 20
    print("="*placeholder+"Total Metrics"+"="*placeholder)
    print("CSR:\t", f"{csr[-1]:.3f}")
    print("ISR:\t", f"{isr[-1]:.3f}")
    print("SSR:\t", f"{ssr[-1]:.3f}")
    
    print("-"*placeholder_2+"Constraints-categorized Results"+"-"*placeholder_2)
    for i, item in enumerate(csr):
        #print(C_LIST[i], f"{item:.3f}", sep=":\t")
        print("{0:15} {1:.3f}".format(C_LIST[i]+":", item))
    
    print("-"*placeholder_2+"Instructions-categoried Results"+"-"*placeholder_2)
    print("{0:15} {1:.3f}".format("Aligned:", isr[0]))
    print("{0:15} {1:.3f}".format("Misaligned:", isr[1]))
    print("{0:15} {1:.3f}".format("Total:", isr[-1]))
    
    print("-"*placeholder_2+"Sessions-categoried Results"+"-"*placeholder_2)
    #print(ssr)
    ssr_dep = ssr[0:6]
    ssr_para = ssr[6:12]
    #print(ssr_dep)
    #print(ssr_para)
    print("Multi-turn Dependent")
    for i in range(0, 5):
        print("\tR{0}:\t{1:.3f}".format(i+1, ssr_dep[i]))
    print("\tAvg:\t{:.3f}".format(ssr_dep[-1]))
    print("Multi-turn Parallel")
    for i in range(0, 5):
        print("\tR{0}:\t{1:.3f}".format(i, ssr_para[i]))
    print("\tAvg:\t{:.3f}".format(ssr_para[-1]))
    print("{0:15} {1:.3f}".format("Total:", ssr[-1]))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model_name", type=str)
    parser.add_argument("--output_dir", type=str, default='./output')
    
    args = parser.parse_args()
    
    infer_model_name = args.infer_model_name
    output_dir = args.output_dir
    read_metrics(infer_model_name, output_dir)
