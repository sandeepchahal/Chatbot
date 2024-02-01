
from openai import OpenAI
import os
import pandas as pd
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")
pd.options.mode.copy_on_write = True

OPEN_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(
    api_key= OPEN_API_KEY
)

maximum_tokens = 300
class MyEmbedding:
    
    def _remove_newlines(self,serie):
        serie = serie.str.replace('\n', ' ')
        serie = serie.str.replace('\\n', ' ')
        serie = serie.str.replace('  ', ' ')
        serie = serie.str.replace('  ', ' ')
        return serie
    
    def _process_files(self,directory_path, output_csv_path):
        ## this function will process all txt files and dump the data into csv by removing some extra spaces,etc.
        texts=[]
        text_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        for text_file in text_files:
            print(f"Processing the file -  {text_file}")
            file_path = os.path.join(directory_path, text_file)
            with open(file_path, 'r') as file:
                text = file.read()
                texts.append(text)

        # # Create a dataframe from the list of texts
        df = pd.DataFrame(texts, columns = ['text'])
        df['text'] = self._remove_newlines(df.text)
        df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
        df.to_csv(output_csv_path)
        print("CSV file is created")
    
    def _split_sentence(self, row, max_tokens = maximum_tokens):
        sentences = row.split('. ')
        n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences] # no of token for each sentence
        chunks = []
        tokens_so_far = 0
        chunk = []

        for sentence, token in zip(sentences, n_tokens):
            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0
            if token > max_tokens:
                continue
            chunk.append(sentence)
            tokens_so_far += token + 1
        return chunks
    
    def createEmbedding(self,directory_path, output_csv_path,model="text-embedding-ada-002"):

        try:
            ## 1. process the files into csv
            self._process_files(directory_path=directory_path, output_csv_path= output_csv_path)
            df = pd.read_csv(output_csv_path)
            df.head()
            ## 2. create tokens 
            ## check if row token in greater than max tokens
            ## if yes, split into chunks and add in new row
            shortened = []
            for row in df.iterrows():
                if row[1]['text'] is None: ## 0th row is column names
                    continue
                if row[1]['n_tokens'] > maximum_tokens:
                    shortened += self._split_sentence(row[1]['text'])
                else:
                    shortened.append( row[1]['text'] )  
            df = pd.DataFrame(shortened, columns = ['text']) # update column text with truncate sentence
            df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x))) # update the token for each row
            print("Sentence split done")

            df['embeddings'] = df.text.apply(lambda x: client.embeddings.create(input=x,model=model).data[0].embedding)
            df.to_csv(directory_path+"embeddings.csv")
            return True
        except:
             raise Exception("an error occurred")
    
    