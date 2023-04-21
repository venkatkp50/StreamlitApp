from summarizer import Summarizer,TransformerSummarizer

header =[]
berttext = []
para = []
BERT_MAX_TOKEN = 512

bert_model = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)

def getBERTSummary(filecount,summarizationFor,std_text,max_abstract_token_size,max_sent_size):
    print('Inside BERT file ................')
    for line in std_text:
        if len(line) > 1:
            if len(line) < 100:
                header.append(line)
            else:
                para.append(line)

    print('BERT .....tot_words_ref =',max_abstract_token_size,'BERT_MAX_TOKEN=',BERT_MAX_TOKEN)
    if max_abstract_token_size > BERT_MAX_TOKEN:
        for parabody in para:
            berttext.append(bert_model(body=parabody,max_length=100))
            berttext = Summarizer(body=parabody,max_length=max_abstract_token_size,num_sentences=max_sent_size)
    else:
        for line in std_text:
            para.append(line) 
        berttext = ''.join( lines for lines in para) 
        berttext = bert_model(body=berttext,max_length=max_abstract_token_size,num_sentences=max_sent_size)
    print('Completed BERT Text..............')
    bert_text = ''.join( lines for lines in berttext) 
    return bert_text