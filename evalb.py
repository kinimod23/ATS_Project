class evalb():
    def __init__(self, parsed, gold):
        self.parsed = self.make_comparable(self.preprocess(parsed))
        self.gold = self.make_comparable(self.preprocess(gold))
        
    def preprocess(self, sentences):
        sentences_formatted =[]
        
        for sent in sentences:
            print('_')
            
            sent = sent.split()
            if 'TOP' in sent[0]:
                sent = sent[1:]
                sent[-1] = sent[-1][:-1]
            elif '(' == sent[0]:
                sent[-2] = sent[-2]+sent[-1]
                sent = sent[1:]
                
            sent_formatted = []
            index = 0
            leaveCount = 0
            while index < len(sent):
                if '-NONE-' in sent[index]:
                    if sent[index+1].count(')') > 1:
                        sent[index-1] = sent[index-1]+(sent[index+1].count(')')-1)*')'
                    sent.remove(sent[index])
                    sent.remove(sent[index])
                index += 1
            index=0
            while index < len(sent):           
                if sent[index][0] == '(' and sent[index+1][-1] == ')':
                    sent_formatted.append(' '.join([sent[index][0], str(leaveCount), sent[index][1:], sent[index+1]]))
                    index += 1
                    leaveCount += 1
                else:  
                    sent_formatted.append(' '.join([sent[index][0], str(leaveCount), sent[index][1:]]))
                index += 1
            sentences_formatted.append(sent_formatted)
            #print('_')
            #print(sentences_formatted)
        return sentences_formatted
    
    def make_comparable(self, sentences):
        sentences_formatted = []
        for sent in sentences:
            result = []
            for index, token in enumerate(sent):
                count1 = 0
                if token[0] == '(':
                    count1 += 1
                    count2 = count1
                    for index2, token2 in enumerate(sent[index:]):
                        #print(token, token2, count1, count2)
                        if token2[0] == '(':
                            count2 += 1
                        if token2[-1] == ')':
                            count2 -= token2.count(')')
                        if count1 >= count2:
                            result.append((token[4:].split()[0], (int(token[2]), int(token2.split()[1])+1)))
                            break
                elif token[-1] == ')':
                    count1 -= token.count(')')
            sentences_formatted.append(result)
        return sentences_formatted
        
