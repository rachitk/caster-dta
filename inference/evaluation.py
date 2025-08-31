import torch
import pandas as pd
import os

from dataset.dual_dataset import PMD_DataLoader
from explanation.explain_wrapper import DTAModelExplainer

from torch_geometric.utils import to_nested_tensor


import ipdb


def run_model_on_dataset(model, dataset, device, max_batch_size=8,
                         do_gnn_explainer=True,
                         save_progressively_instead=False,
                         save_per_batch=1000,
                         output_dir=None):
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    elif save_progressively_instead:
        raise ValueError("No output directory specified - cannot save progressively", flush=True)


    # Make the DataLoader from this dataset and return the affinity + attention results
    # performing some parsing to get the attention matrices and lengths of the proteins and molecules
    # Note that shuffle=False doesn't work properly here (still shuffles)
    # so we'll just have to parse the results later
    dataloader = PMD_DataLoader(dataset, max_batch_size=max_batch_size, shuffle=False)
    n_pairs = len(dataset)

    explainer = DTAModelExplainer(model, n_epochs=10)

    proc_so_far = 0

    all_preds = []
    full_prot_drug_attns, full_drug_prot_attns = [], []

    all_prot_ids, all_drug_ids = [], []
    all_prot_lens, all_drug_lens = [], []

    all_prot_explain, all_drug_explain = [], []

    for i, (prot_g, mol_g, _) in enumerate(dataloader):
        prot_g = prot_g.to(device)
        mol_g = mol_g.to(device)

        with torch.no_grad():
            prot_ids = prot_g.protein_id
            # prot_seqs = prot_g.protein_sequence
            mol_ids = mol_g.molecule_id
            # mol_smiles = mol_g.molecule_smiles
            all_prot_ids.extend(prot_ids)
            all_drug_ids.extend(mol_ids)

            # Get the predictions and attention matrices
            pred, attn_mats = model.forward_with_graphs(prot_g, mol_g)

            # Unscale the predictions
            pred = dataset.unscale_target(pred)
            all_preds.append(pred)

            # Save the lengths of the proteins and molecules
            # (number of nodes in each graph in the batch)
            prot_lens = prot_g.ptr.diff()
            drug_lens = mol_g.ptr.diff()

            # Save the attention matrices (averaged to get protein and atom weights)
            # normalized per-protein and per-molecule
            prot_drug_attns = attn_mats[0][1]
            drug_prot_attns = attn_mats[0][0]

            list_pd_attn, list_dp_attn = [], []

            for pd_attn, dp_attn, plen, dlen in zip(prot_drug_attns, drug_prot_attns, prot_lens, drug_lens):
                list_pd_attn.append(pd_attn[:dlen, :plen].cpu())
                list_dp_attn.append(dp_attn[:plen, :dlen].cpu())

            full_prot_drug_attns.extend(list_pd_attn)
            full_drug_prot_attns.extend(list_dp_attn)

            all_prot_lens.extend(prot_lens.tolist())
            all_drug_lens.extend(drug_lens.tolist())

        if(do_gnn_explainer):
            prot_g_dict, mol_g_dict = model._graphs_to_dicts(prot_g, mol_g)
            explanations = explainer.explain_model(prot_g_dict, mol_g_dict)

            prot_exps = to_nested_tensor(explanations['protein'].node_mask, ptr=prot_g.ptr).cpu()
            drug_exps = to_nested_tensor(explanations['molecule'].node_mask, ptr=mol_g.ptr).cpu()

            all_prot_explain.extend([x[:,-1].softmax(dim=0).numpy() for x in prot_exps])
            all_drug_explain.extend([x[:,-1].softmax(dim=0).numpy() for x in drug_exps])


        # Print progress
        n_batch = len(prot_g.ptr) - 1
        proc_so_far += n_batch
        print(f"Processed {proc_so_far}/{n_pairs} pairs", end="\r", flush=True)

        if save_progressively_instead and output_dir is not None:
            if i % save_per_batch == 0:
                parsed_df = parse_df_from_input(all_preds, all_prot_ids, all_drug_ids, full_prot_drug_attns, full_drug_prot_attns,
                                                all_prot_explain, all_drug_explain, all_prot_lens, all_drug_lens)
                parsed_df.to_pickle(os.path.join(output_dir, f"results_{i}.pkl"))

                # Reset the lists for the next batch
                all_preds = []
                full_prot_drug_attns, full_drug_prot_attns = [], []
                all_prot_ids, all_drug_ids = [], []
                all_prot_lens, all_drug_lens = [], []
                all_prot_explain, all_drug_explain = [], []

                del parsed_df

                print(f"Completed processing {i} batches", flush=True)

    print(f"Completed processing {n_pairs} pairs", flush=True)


    if not save_progressively_instead:
        # Final results will be returned for user and fully saved to file
        parsed_df = parse_df_from_input(all_preds, all_prot_ids, all_drug_ids, full_prot_drug_attns, full_drug_prot_attns,
                                        all_prot_explain, all_drug_explain, all_prot_lens, all_drug_lens)
        
        if output_dir is not None:
            parsed_df.to_pickle(os.path.join(output_dir, f"results_full.pkl"))
    else:
        # Save the last batch to file
        parsed_df = parse_df_from_input(all_preds, all_prot_ids, all_drug_ids, full_prot_drug_attns, full_drug_prot_attns,
                                        all_prot_explain, all_drug_explain, all_prot_lens, all_drug_lens)
        
        if output_dir is not None:
            parsed_df.to_pickle(os.path.join(output_dir, f"results_{i+1}.pkl"))
        
        print(f"Completed processing {i+1} batches", flush=True)

    
    return parsed_df



def parse_df_from_input(all_preds, all_prot_ids, all_drug_ids, full_prot_drug_attns, full_drug_prot_attns,
                        all_prot_explain, all_drug_explain, all_prot_lens, all_drug_lens,
                        include_full_matrix=False):
    with torch.no_grad():
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        parsed_prot_attns, parsed_drug_attns = [], []
        parsed_max_prot_attns, parsed_max_drug_attns = [], []

        for prot_drug_attn, drug_prot_attn in zip(full_prot_drug_attns, full_drug_prot_attns):
            parsed_prot_attns.append(prot_drug_attn.mean(dim=0).numpy())
            parsed_drug_attns.append(drug_prot_attn.mean(dim=0).numpy())

            parsed_max_prot_attns.append(prot_drug_attn.max(dim=0).values.numpy())
            parsed_max_drug_attns.append(drug_prot_attn.max(dim=0).values.numpy())

        parsed_df = pd.DataFrame.from_dict({
            "protein_id": all_prot_ids,
            "molecule_id": all_drug_ids,
        })

        if not all_prot_explain:
            all_prot_explain = [None] * len(all_prot_ids)
            all_drug_explain = [None] * len(all_drug_ids)

        parsed_df['affinity_score'] = all_preds
        parsed_df['protein_attention'] = parsed_prot_attns
        parsed_df['molecule_attention'] = parsed_drug_attns
        parsed_df['max_protein_attention'] = parsed_max_prot_attns
        parsed_df['max_molecule_attention'] = parsed_max_drug_attns
        parsed_df['protein_explanation'] = all_prot_explain
        parsed_df['molecule_explanation'] = all_drug_explain
        parsed_df['max_protein_explanation'] = all_prot_explain
        parsed_df['max_molecule_explanation'] = all_drug_explain
        parsed_df['protein_len'] = all_prot_lens
        parsed_df['molecule_len'] = all_drug_lens

        if include_full_matrix:
            parsed_df['prot_mol_attention'] = [x.numpy() for x in full_prot_drug_attns]
            parsed_df['mol_prot_attention'] = [x.numpy() for x in full_drug_prot_attns]

    return parsed_df