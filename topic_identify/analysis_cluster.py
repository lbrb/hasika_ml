import pandas as pd
from topic_identify.pr_curve import HasikaPrCurve


class AnalysisCluster:
    def process(self):
        path = 'cluster_news_人工聚类_930cluster.xlsx'
        df = pd.read_excel(path)
        ai_dict = {}
        rengong_dict = {}
        for index, item in df.iterrows():
            ai = item['聚类']
            rengong = item['人工']
            title = item['新闻标题']

            if ai not in ai_dict.keys():
                ai_dict[ai] = [index]
            else:
                ai_dict[ai].append(index)

            if rengong not in rengong_dict.keys():
                rengong_dict[rengong] = [index]
            else:
                rengong_dict[rengong].append(index)
        ai_list = list(ai_dict.values())
        rengong_list = list(rengong_dict.values())

        pr_curve = HasikaPrCurve()
        p, r = pr_curve.calc_pr(rengong_list, ai_list)
        print(p, r)


a = AnalysisCluster()
a.process()
