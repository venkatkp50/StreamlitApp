from summarizer import Summarizer,TransformerSummarizer

header =[]
gpt2text = []
para = []
GPT2_MAX_TOKEN = 1024

GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="distilgpt2")

def getGPT2Summary(filecount,summarizationFor,full_text,max_abstract_token_size,max_sent_size):
    if max_abstract_token_size > GPT2_MAX_TOKEN:  
        for line in full_text:
            if len(line) > 1:
                if len(line) < 100:
                    header.append(line)
                else:
                    para.append(line)
        for parabody in para:
            gpt2text.append(GPT2_model(body=parabody, max_length=100))                           
            gpt2text_full = ''.join(text for text in gpt2text)
            gpt2_summary = GPT2_model(body=gpt2text_full, max_length=max_abstract_token_size,num_sentences=max_sent_size)
    else:
            for line in full_text:
                para.append(line)
            gpt2text = ''.join( lines for lines in para) 
            gpt2text = GPT2_model(body=gpt2text, max_length=max_abstract_token_size,num_sentences=max_sent_size)
    
    gpt2_summary = ''.join( lines for lines in gpt2text)
    
    return gpt2_summary
