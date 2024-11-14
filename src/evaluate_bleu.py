import evaluate
folder = "results/c_trained_infill"
src = "s2sft_wmt14_en_de_detokenize.txt"
infill_source =  "infill_wmt_en_de.txt"
metric_trg = 'data/wmt14_en_de/test.de'
length_src = 'test_data.txt'
# metric_trg = 'test_data.txt'
# src = "s2sft_wmt14_en_de_detokenize.txt"
src = f's2sft_wmt14_en_de_t5_sample_length_5.txt'

def get_score_bleu():
    bleu_score = evaluate.load('bleu')
    with open(f"{folder}/{src}") as pred_file:
        with open(metric_trg) as target_file:
    # with open(f"{folder}/{infill_source}") as pred_file:
        # with open(f"{folder}/{length_src}") as target_file:
            pred = pred_file.readlines()
            pred_list = []
            for idx,p in enumerate(pred):
                if idx%2 == 1:
                    pred_list.append(p.strip())
                    
            # pred_list = [p.strip() for p in pred]
            target = target_file.readlines()
            target_list = [[t.strip()] for t in target]
            score = bleu_score.compute(predictions=pred_list, references=target_list)
            print(score)
            
def get_score_rouge():
    rouge_score = evaluate.load('rouge')
    with open(f"{folder}/{src}") as pred_file:
        with open(metric_trg) as target_file:
    # with open(f"{folder}/{infill_source}") as pred_file:
        # with open(f"{folder}/{length_src}") as target_file:
            pred = pred_file.readlines()
            pred_list = [p.strip() for p in pred]
            target = target_file.readlines()
            target_list = [[t.strip()] for t in target]
            print(pred_list[0])
            print(target_list[0])
            score = rouge_score.compute(predictions=pred_list, references=target_list)
            print(score)
def get_length_score():
    score = 0
    with open(f"{folder}/{infill_source}") as pred_file:
        with open(f"{folder}/{length_src}") as target_file:
            pred = pred_file.readlines()
            pred_list = [p.strip() for p in pred]
            target = target_file.readlines()
            target_list = [t.strip() for t in target]
            first_pred = []
            first_trg = []
            for pre in pred_list:
                pre = pre.split(" ")
                first_pred.append(pre[0])
            for trgg in target_list:
                trgg = trgg.split(" ")
                first_trg.append(trgg[0])
            print(first_pred[584])
            print(first_trg[315])
            # exit()
            for p,t in zip(first_pred,first_trg):
                if p==t:
                    score += 1
    print(score/len(first_trg))
    

def sim_score():
    from sentence_transformers import SentenceTransformer
    import math
    from numpy.linalg import norm

    # se = SentenceTransformer('aari1995/German_Semantic_STS_V2')
    se = SentenceTransformer('danielheinz/e5-base-sts-en-de')
    se = SentenceTransformer('jinaai/jina-embeddings-v2-base-de', trust_remote_code=True)
    # cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))

    # embeddings = se.encode([
    # ': Noch mehr Sicherheit für',
    # 'Am Dienstag ist in Gutach'
    # ])
    # print(cos_sim(embeddings[0], embeddings[1]))
    # exit()
    with open(f"{folder}/{src}") as pred_file:
        with open(metric_trg) as target_file:
    # with open(f"{folder}/{infill_source}") as pred_file:
        # with open(f"{folder}/{length_src}") as target_file:
            pred = pred_file.readlines()
            pred_list = [p.strip() for p in pred]
            target = target_file.readlines()
            target_list = [t.strip() for t in target]
            trg_query = []
            src_trg_query = []
            for p,t in zip(pred_list,target_list):
                trg_query.append(t)
                src_trg_query.append(p)
            trg_emb = se.encode(trg_query)
            src_trg_emb = se.encode(src_trg_query)
            scores = []
            for t_emb, s_emb in zip(trg_emb, src_trg_emb):
                score = se.similarity(t_emb,s_emb)
                print(score)
                scores.append(score)
            print(sum(score)/len(score))

                
# sim_score()
# get_length_score()
get_score_bleu()
# get_score_rouge()

