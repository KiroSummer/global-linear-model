#!/usr/bin/python
#coding=utf-8

import datetime

"""类名称：sentence
   类功能：存储一个句子的组成内容，包括：所有的word、tag、以及每一个word对应单个字wordchars"""
class sentence:
    def __init__(self):
        self.word = []
        self.tag = []
        self.wordchars = []

"""类名称：dataset
   类功能：数据集类，用来存储一个conll格式的文件，以数组的形式存储所有的sentence"""
class dataset:
    def __init__(self):
        self.sentences = []
        self.name = ""
        self.total_word_count = 0

    def open_file(self, inputfile):
        self.inputfile = open(inputfile, mode = 'r')
        self.name = inputfile

    def close_file(self):
        self.inputfile.close()

    def read_data(self, sentenceLen):
        wordCount = 0
        sentenceCount = 0
        sen = sentence()
        for s in self.inputfile:
            if(s == '\r\n' or s == '\n'):  #空白行的判断，仅仅只有换行符==空白行==新的一个句子
                sentenceCount += 1
                self.sentences.append(sen)
                sen = sentence()
                if(sentenceLen !=-1 and sentenceCount >= sentenceLen):
                    break
                continue
            list_s = s.split('\t')  #以\t来作为分隔符
            str_word = list_s[1].decode('utf-8')  #decode('utf8')的原因是为了更好的处理wordchars(python2.7需要这样做，3.0以上不需要)
            str_tag = list_s[3]
            list_wordchars = list(str_word)
            sen.word.append(str_word)
            sen.tag.append(str_tag)
            sen.wordchars.append(list_wordchars)
            wordCount += 1
        self.total_word_count = wordCount  #统计word的个数
        self.total_sentence_count = len(self.sentences)  #统计sentence的个数
        print(self.name + " contains " + str(self.total_sentence_count) + " sentences")
        print(self.name + " contains " + str(self.total_word_count) + " words")

"""类名称：global_linear_model
   类功能：提供global linear model的相关功能函数：训练，评估"""
class global_linear_model:
    def __init__(self):
        self.feature = dict()  #存储训练集得到的特征字典集合
        self.feature_keys = []
        self.feature_values = []
        self.feature_length = 0
        self.tags = dict()  #存储所有的词性tag字典集合
        self.tags_length = 0
        self.v = []  #v
        self.update_times =[]  #更新的时间
        self.w = []  #权重weight
        self.train = dataset()  #训练集
        self.dev = dataset()  #开发集

        self.train.open_file("train.conll")
        #self.train.open_file("./data/train.conll")
        #self.train.open_file("./new_data/train_new.conll")
        #self.train.open_file("zy.txt")
        self.train.read_data(100)
        self.train.close_file()

        self.dev.open_file("dev.conll")
        #self.dev.open_file("./data/dev.conll")
        #self.dev.open_file("./new_data/dev_new.conll")
        self.dev.read_data(10)
        self.dev.close_file()

    def create_feature(self, sentence, pos, ti_left_tag = "NULL"):
        word_count = len(sentence.word)
        wi = sentence.word[pos]
        pos_word_len = len(sentence.word[pos])
        if ti_left_tag == "NULL":
            if pos == 0:
                ti_left_tag == "START"
            else:
                ti_left_tag = sentence.tag[pos - 1]
        else:
            ti_left_tag = ti_left_tag
        if(pos == 0):
            wi_left_word = "$$"
            wi_left_word_last_c = "$$"
        else:
            wi_left_word = sentence.word[pos-1]
            wi_left_word_last_c = sentence.wordchars[pos-1][len(sentence.word[pos-1])-1]
        if(pos == word_count-1):
            wi_right_word = "##"
            wi_right_word_first_c = "##"
        else:
            wi_right_word = sentence.word[pos+1]
            wi_right_word_first_c = sentence.wordchars[pos+1][0]
        wi_last_c = sentence.wordchars[pos][pos_word_len - 1]
        wi_first_c = sentence.wordchars[pos][0]
        f = []
        f.append("01:" + ti_left_tag)
        f.append("02:" + wi)
        f.append("03:" + wi_left_word)
        f.append("04:" + wi_right_word)
        f.append("05:" + wi + '*' + wi_left_word_last_c)
        f.append("06:" + wi + '*' + wi_right_word_first_c)
        f.append("07:" + wi_first_c)
        f.append("08:" + wi_last_c)
        for i in range(1, pos_word_len - 1):
            wi_kth_c = sentence.wordchars[pos][i]
            f.append("09:" + wi_kth_c)
            f.append("10:" + wi_first_c + "*" + wi_kth_c)
            f.append("11:" + wi_last_c + "*" + wi_kth_c)
        for i in range(0, pos_word_len - 1):
            wi_kth_c = sentence.wordchars[pos][i]
            wi_kth_next_c = sentence.wordchars[pos][i + 1]
            if(wi_kth_c == wi_kth_next_c):
                f.append("13:" + wi_kth_c + "*" + "consecutive")
        if(pos_word_len == 1):
            f.append("12:" + wi + "*" + wi_left_word_last_c + "*" + wi_right_word_first_c)
        for i in range(0, pos_word_len):
            if(i >= 4):
                break
            f.append("14:" + sentence.word[pos][0:(i + 1)])
            f.append("15:" + sentence.word[pos][-(i + 1)::])
        return f

    def create_feature_space(self):
        feature_index = 0
        tag_index = 0
        for s in self.train.sentences:
            for p in range(0, len(s.word)):
                f = self.create_feature(s, p)
                for feature in f:
                    if (feature in self.feature):
                        pass
                    else:
                        self.feature[feature] = feature_index
                        feature_index += 1
                if(s.tag[p] in self.tags):
                    pass
                else:
                    self.tags[s.tag[p]] = tag_index
                    tag_index += 1
        self.feature_length = len(self.feature)
        self.tags_length = len(self.tags)
        self.tags_keys = list(self.tags.keys())
        self.w = [0]*(self.feature_length * self.tags_length)
        self.v = [0]*(self.feature_length * self.tags_length)
        self.update_times = [0]*(self.feature_length * self.tags_length)
        self.feature_keys = list(self.feature.keys())
        self.feature_values = list(self.feature.values())
        print("the total number of features is " + str(self.feature_length))
        print("the total number of tags is " + str(self.tags_length))
        print(self.tags)

    def dot(self, f_id, offset):
        score = 0
        for f in f_id:
            score += self.w[offset + f]
        return score

    def dot_v(self, f_id, offset):
        score = 0
        for f in f_id:
            score += self.v[offset + f]
        return score

    def get_feature_id(self, fv):
        fv_id = []
        for feature in fv:
            if(feature in self.feature):
                fv_id.append(self.feature[feature])
        return fv_id;

    def max_tag_sequence(self, sentence, use_v = False):
        list_sentence_tag = []  #用来保存路径
        tag_sequence = []
        last_score = dict()
        current_score = dict()
        last_tag = dict()
        current_tag = dict()
        maxscore = float("-Inf")
        maxtag = "NULL"
        sentence_length = len(sentence.word)
        #第一个词
        feature = self.create_feature(sentence, 0, "NULL")
        feature_id = self.get_feature_id(feature)
        for t in self.tags:
            if(use_v == True):
                tempscore = self.dot_v(feature_id, self.feature_length * self.tags[t])
            else:
                tempscore = self.dot(feature_id, self.feature_length * self.tags[t])
            current_score[t] = tempscore
            current_tag[t] = "START"
        list_sentence_tag.append(current_tag)
        #如果只有一个词
        if(sentence_length == 1):
            maxscore = float("-Inf")
            maxtag = "NULL"
            for tag in current_score.keys():
                c_score = current_score[tag]
                if(c_score > maxscore):
                    maxscore = c_score
                    maxtag = tag
        ith_max_tag = maxtag
        last_score = current_score
        current_score = dict()
        last_tag = current_tag
        current_tag = dict()
        #后几个词
        for pos in range(1, sentence_length):
            maxscore = float("-Inf")
            maxtag = "NULL"
            current_score = dict()
            current_tag = dict()
            for tag in self.tags:
                maxscore = float("-Inf")
                maxtag = "NULL"
                for t in last_score.keys():
                    feature = self.create_feature(sentence, pos, t)
                    feature_id = self.get_feature_id(feature)
                    if(use_v == True):
                        offset = self.feature_length * self.tags[tag]
                        tempscore = self.dot_v(feature_id, offset) + last_score[t]
                    else:
                        offset = self.feature_length * self.tags[tag]
                        tempscore = self.dot(feature_id, offset) + last_score[t]
                    if(tempscore > maxscore):
                        maxscore = tempscore
                        maxtag = t
                current_score[tag] = maxscore
                current_tag[tag] = maxtag
            #找到上一个词最大的tag
            maxscore = float("-Inf")
            maxtag = "NULL"
            for tag in current_score.keys():
                if(current_score[tag] > maxscore):
                    maxscore = current_score[tag]
                    maxtag = tag
            #tag_sequence.append(current_tag[maxtag])
            last_score = current_score
            last_tag = current_tag
            ith_max_tag = maxtag
            list_sentence_tag.append(current_tag)
        tag_sequence.append(ith_max_tag)
        list_sentence_tag.reverse()
        #print list_sentence_tag[-2]
        for i in range(len(list_sentence_tag) - 1):
            ith_max_tag = list_sentence_tag[i][ith_max_tag]
            tag_sequence.append(ith_max_tag)
        tag_sequence.reverse()
        #print len(sentence.word)
        #print len(tag_sequence)
        #print tag_sequence
        #print sentence.tag
        return tag_sequence

    def update_v(self, index, update_times, last_w_value):
        last_update_times = self.update_times[index]    #上一次更新所在的次数
        current_update_times = update_times    #本次更新所在的次数
        self.update_times[index] = update_times
        self.v[index] += (current_update_times - last_update_times -1 )*last_w_value + self.w[index]

    def update_weight(self, s, max_tag_sequence, correct_tag_sequence, update_times):
        feature = self.create_feature(s, 0, "NULL")
        feature_id = self.get_feature_id(feature)
        maxtag_id = self.tags[max_tag_sequence[0]]
        for i in feature_id:
            offset = self.feature_length * maxtag_id
            index = offset + i
            last_w_value = self.w[index]    #更新前的权重
            self.w[index] -= 1
            self.update_v(index, update_times, last_w_value)
        for i in range(1, len(max_tag_sequence)):
            feature = self.create_feature(s, i, max_tag_sequence[i - 1])
            feature_id = self.get_feature_id(feature)
            maxtag_id = self.tags[max_tag_sequence[i]]
            for i in feature_id:
                offset = self.feature_length * maxtag_id
                index = offset + i
                last_w_value = self.w[index]    #更新前的权重
                self.w[index] -= 1
                self.update_v(index, update_times, last_w_value)
        feature = self.create_feature(s, 0, "NULL")
        feature_id = self.get_feature_id(feature)
        correct_id = self.tags[correct_tag_sequence[0]]
        for i in feature_id:
            offset = self.feature_length * correct_id
            index = offset + i
            last_w_value = self.w[index]    #更新前的权重
            self.w[index] += 1
            self.update_v(index, update_times, last_w_value)
        for i in range(1, len(correct_tag_sequence)):
            feature = self.create_feature(s, i, correct_tag_sequence[i - 1])
            feature_id = self.get_feature_id(feature)
            correcttag_id = self.tags[correct_tag_sequence[i]]
            for i in feature_id:
                offset = self.feature_length * correcttag_id
                index = offset + i
                last_w_value = self.w[index]    #更新前的权重
                self.w[index] += 1
                self.update_v(index, update_times, last_w_value)

    """函数名称：perceptron_online_training
       函数功能：对global linear model进行训练
       函数返回：无"""
    def perceptron_online_training(self):
        max_train_precision, max_dev_precision, update_times = 0.0, 0.0, 0
        for iterator in range(0, 20):  #进行20次迭代
            print("iterator " + str(iterator))
            for s in self.train.sentences:
                max_tag_sequence = self.max_tag_sequence(s)  #list
                correct_tag_sequence = s.tag  #list
                if(max_tag_sequence != correct_tag_sequence):
                    update_times += 1
                    self.update_weight(s, max_tag_sequence, correct_tag_sequence, update_times)
            #本次迭代完成
            current_update_times = update_times    #本次更新所在的次数
            for i in range(len(self.v)):
                last_w_value = self.w[i]
                last_update_times = self.update_times[i]    #上一次更新所在的次数
                if(current_update_times != last_update_times):
                    self.update_times[i] = current_update_times
                    self.v[i] += (current_update_times - last_update_times - 1)*last_w_value + self.w[i]

            #self.save_model(iterator)
            #进行评估
            dev_iterator, dev_c, dev_count, dev_precision = self.evaluate(self.dev, iterator)
            #保存概率最大的情况
            if(dev_precision > (max_dev_precision + 1e-10)):
                max_dev_precision, max_dev_iterator, max_dev_c, max_dev_count = dev_precision, dev_iterator, dev_c, dev_count
        print("Conclusion:")
        print("\t"+self.dev.name + " iterator: "+str(max_dev_iterator)+"\t"+str(max_dev_c)+" / "+str(max_dev_count) + " = " +str(max_dev_precision))

    def save_model(self, iterator):
        fmodel = open("linearmodel.lm"+str(iterator), mode='w')
        for feature_id in self.feature_values:
            feature = self.feature_keys[feature_id]
            left_feature = feature.split(':')[0] + ':'
            right_feature = '*' + feature.split(':')[1]
            for tag in self.tags:
                tag_id = self.tags[tag]
                entire_feature = left_feature + tag + right_feature
                w = self.w[tag_id * self.feature_length + feature_id]
                if(w != 0):
                    fmodel.write(entire_feature.encode('utf-8') + '\t' + str(w) + '\n')
        fmodel.close()

    """函数名称：evaluate
       函数功能：根据开发集dev，测试global linear model训练得到的成果
                 输出正确率
       函数返回：迭代次数，正确的tag的个数，所有的word的个数，准确率"""
    def evaluate(self, dataset, iterator):
       c = 0  #记录标注正确的tag数量
       for s in dataset.sentences:
           max_tag_sequence = self.max_tag_sequence(s, True)  #使用v进行评估，返回得分最高的tag序列
           correct_tag_sequence = s.tag
           for i in range(len(max_tag_sequence)):  #比较每一个tag，是否相等？
               if(max_tag_sequence[i] == correct_tag_sequence[i]):
                   c += 1
       accuracy = 1.0 * c / dataset.total_word_count
       print(dataset.name + "\tprecision is " + str(c) + " / " + str(dataset.total_word_count) + " = " + str(accuracy))
       return iterator, c, dataset.total_word_count, accuracy


################################ main #####################################
if __name__ == '__main__':
    starttime = datetime.datetime.now()
    glm = global_linear_model()
    glm.create_feature_space()  #创建特征空间
    glm.perceptron_online_training()  #global linear model perceptron online training
    endtime = datetime.datetime.now()
    print("executing time is "+str((endtime-starttime).seconds)+" s")
