from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class VIdeoRoIEval:
    def __init__(self, anno):
        self.anno = anno
        self.videos = (self.anno.keys())
        self.videos = sorted(self.videos, key=lambda x:x[:-4])
        self.vidToEval = {}
        self.eval = {}
    def evaluate(self):
        gts = {}
        res = {}
        for ViDId, video_name in enumerate(self.videos):
            gts[ViDId] = [{'caption':self.anno[video_name]['groundtruth']}]
            res[ViDId] = [{'caption':self.anno[video_name]['pred']}]
        # print(gts)
        # print(res)
        

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setVidToEvalVids(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setVidToEvalVids(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalVids()

    def setEval(self, score, method):
        self.eval[method] = score

    def setVidToEvalVids(self, scores, vidIds, method):
        for vidId, score in zip(vidIds, scores):
            if not vidId in self.vidToEval:
                self.vidToEval[vidId] = {}
                self.vidToEval[vidId]["video_id"] = vidId
            self.vidToEval[vidId][method] = score

    def setEvalVids(self):
        self.evalVids = [eval for vidId, eval in self.vidToEval.items()]
    
if __name__ == '__main__':
    anno = {'test.mp4':{'groundtruth':'The woman walks to the bed and sits next to man.',\
                        'generated':'The woman in red dress walks toward the man.'}}
    VideoRoIBench = VIdeoRoIEval(anno)
    VideoRoIBench.evaluate()
    for metric, score in VideoRoIBench.eval.items():
        print(f'{metric}: {score:.3f}')