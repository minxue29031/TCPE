import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class TextCluster:
    def __init__(self, embedding_model: str, max_length: int, threshold: float):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = embedding_model
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.max_length = max_length
        self.threshold = threshold

    def _load_model_and_tokenizer(self):
        model = AutoModel.from_pretrained(
            self.embedding_model,
            trust_remote_code=True,
            unpad_inputs=True,
            use_memory_efficient_attention=True,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        return model, tokenizer

    def embed_texts(self, texts):
        inputs = self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            with torch.inference_mode():
                outputs = self.model(**inputs.to(self.device)).last_hidden_state[:, 0]  
        return F.normalize(outputs, p=2, dim=1).cpu().numpy()

    def update_groups(self, new_df: pd.DataFrame, ori_df: pd.DataFrame):
        max_group_number = self._get_max_group_number(ori_df['group'].unique())
        all_groups = new_df['group'].unique()

        for group in all_groups:            
            new_samples = new_df[new_df['group'] == group]

            if new_samples.empty:
                continue

            sample = new_samples.sample(1)
            sample_brief = sample['brief_description'].values[0]
            ori_embeddings = self.embed_texts(ori_df['brief_description'].tolist())
            sample_embedding = self.embed_texts([sample_brief])
            cosine_sim = cosine_similarity(sample_embedding, ori_embeddings)

            similar_indices = np.where(cosine_sim[0] > self.threshold)[0]

            if len(similar_indices) > 0:
                new_group = ori_df.iloc[similar_indices[0]]['group']
                for i in range(1, 6):
                    new_df.loc[new_df['group'] == group, f'rename_group_{i}'] = new_group
            else:
                max_group_number += 1
                new_group_name = str(max_group_number)
                for i in range(1, 6):
                    new_df.loc[new_df['group'] == group, f'rename_group_{i}'] = new_group_name

        return new_df

    def most_common_or_empty(self, row):
        groups = [row[f'rename_group_{i}'] for i in range(1, 6)]
        group_count = Counter(groups)
        most_common_group, count = group_count.most_common(1)[0] if group_count else (None, 0)

        return most_common_group if count >= 4 else "issue"

    def process_clusters(self, ori_csv: str, new_csv: str, update_process_csv: str, updated_cluster_csv: str, update_source_cluster_csv: str):
        ori_df = pd.read_csv(ori_csv)
        new_df = pd.read_csv(new_csv)

        print("ori DataFrame column:", ori_df.columns)
        print("new DataFrame column:", new_df.columns)

        ori_df['brief_description'] = ori_df['brief_description'].fillna('')
        new_df['brief_description'] = new_df['brief_description'].fillna('')
        new_df['ori_group'] = new_df['group']

        updated_new_df = self.update_groups(new_df, ori_df)
        updated_new_df.drop(columns=['group'], inplace=True)
        updated_new_df['rename_group'] = updated_new_df.apply(self.most_common_or_empty, axis=1)

        updated_new_df.to_csv(update_process_csv, index=False)
        updated_new_df.drop(columns=[f'rename_group_{i}' for i in range(1, 6)], inplace=True)
        updated_new_df.to_csv(updated_cluster_csv, index=False)

        print(f"Files saved as {update_process_csv} and {updated_cluster_csv}.")

        # Get new groups and append them to the original file
        max_ori_group = self._get_max_group_number(ori_df['group'].unique())

        new_groups_df = updated_new_df[updated_new_df['rename_group'].notna() & updated_new_df['rename_group'].str.isnumeric()]
        result_indices = new_groups_df.index.tolist()
        print("======NAN index===", result_indices)

        new_groups_df = new_groups_df[new_groups_df['rename_group'].astype(int) > max_ori_group]

        if not new_groups_df.empty:
            new_groups_df_to_append = new_groups_df.drop(columns=['ori_group']).rename(columns={'rename_group': 'group'})
            updated_ori_df = pd.concat([ori_df, new_groups_df_to_append], ignore_index=True)
            updated_ori_df.to_csv(update_source_cluster_csv, index=False)

            print(f"File saved as {update_source_cluster_csv} (there are new groups).")
        else:
            ori_df.to_csv(update_source_cluster_csv, index=False)
            print(f"File saved as {update_source_cluster_csv} (no new groups).")

    @staticmethod
    def _get_max_group_number(groups):
        numeric_groups = []
        for group in groups:
            try:
                numeric_groups.append(int(group))
            except ValueError:
                continue
        return max(numeric_groups) if numeric_groups else 0


if __name__ == "__main__":

    ori_csv_path = 'path_to_cluster_ori.csv'
    new_csv_path = 'path_to_cluster_new.csv' 
    embedding_model="Alibaba-NLP/gte-base-en-v1.5"
    max_length = 6
    threshold = 0.9
    
    update_process_clsuter_path = 'path_to_update_cluster_process.csv'
    updated_cluster_path = 'path_to_updated_current_cluster.csv'
    update_source_cluster_path = 'path_to_update_source_cluster.csv'  

    text_cluster = TextCluster(embedding_model, max_length, threshold)
    text_cluster.process_clusters(
        ori_csv=ori_csv_path,
        new_csv=new_csv_path,
        update_process_csv=update_process_clsuter_path,
        updated_cluster_csv=updated_cluster_path,
        update_source_cluster_csv=update_source_cluster_path
    )
