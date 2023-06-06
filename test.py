# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/4
@Time    : 16:31
@File    : test.py
@Function: XX
@Other: XX
"""
import json
import requests
from tqdm import tqdm


def server_test(sen):
    sess_web = requests.Session()
    # noinspection PyBroadException
    try:
        req = requests.Request('POST', url=url, data=sen.encode("utf-8"))
        prep = sess_web.prepare_request(req)
        results = sess_web.send(prep, stream=False).text
    except Exception as e:
        results = str(e)
    return results


if __name__ == '__main__':
    url = 'http://10.17.107.66:12003/prediction'
    # url = 'http://10.17.107.61:8100'
    text = "废气排放二零二六年底前NOx排放减少3%二零二六年底前SOx排放减少3%二零二六年底前PM排放减少3%GHG排放二零二六年底前范围1排放减少3%二零二六年底前范圉2排放减少3%二零二六年底前范圉3排放减少3%;本集团致力在整个营运过程中善用资源，多管齐下实施节约用水及节能措施，例如鼓励倔员用后关掉灯具及空调、用水后紧闭水龙头及更有效使用纸张。本集团亦非常重视有效使用纸张，推广使用环保纸进行休闲打印及草稿，并张贴节约用纸标志，鼓励偏员更有效地使用纸张。此等措施符合本集团更广泛的可持续发展目标，显示其致力减少环境影响的决心。通过提高倔员的资源节约意识及实施有效的攀措及计划，本集团正操取积极措施促进可持续发展。"
    text = '一、本期业绩预告情况：(一)业绩预告期间。2021年1月1日至2021年12月31日。(二)业绩预告情况。1、经财务部门初步测算，洛阳建龙微纳新材料股份有限公司（以下简称“公司”）预计2021年年度实现归属于母公司所有者的净利润为27,500.00万元至29,000.00万元，与上年同期（法定披露数据）相比，将增加14,762.93万元至16,262.93万元，同比增加115.91%至127.68%。'
    text = '该单位存在无《辐射安全许可证》，从事射线装置使用活动的环境违法行为@@该单位存在无《辐射安全许可证》，从事射线装置使用活动的环境违法行为。'
    # text = '禅意歌者刘珂矣《一袖云》中诉知己…绵柔纯净的女声，将心中的万水千山尽意勾勒于这清素画音中'
    # text = '兴业银行。2022年3月25日，银保监罚决字(2022)22号显示，兴业银行股份有限公司存在漏报贸易融资业务EAST数据等14项违法违规事实，被中国人民银行处罚款350万元。'
    # text = """<table><tr><td>序号 </td><td>企业名称 </td><td>注册资本 </td><td>持股比例 </td><td>子公司级别 </td><td>业务性质 </td></tr><tr><td>1 </td><td>泸州临港产业开发有限公司 </td><td>20,000.00 </td><td>100.00 </td><td>二级 </td><td>房地产 </td></tr><tr><td>2 </td><td>泸州临港工业化建筑科技有限公司 </td><td>10,000.00 </td><td>65.00 </td><td>三级 </td><td>PC制品 </td></tr><tr><td>3 </td><td>泸州临港产业建设有限公司 </td><td>15,000.00 </td><td>100.00 </td><td>二级 </td><td>工程施工 </td></tr><tr><td>4 </td><td>泸州临港诺达建材有限公司 </td><td>1,000.00 </td><td>51.00 </td><td>三级 </td><td>建材 </td></tr><tr><td>5 </td><td>泸州临港工程管理有限公司 </td><td>2,000.00 </td><td>51.00 </td><td>三级 </td><td>服务 </td></tr><tr><td>6 </td><td>宜宾拓戎建筑工程有限公司 </td><td>1,000.00 </td><td>100.00 </td><td>三级 </td><td>建筑 </td></tr><tr><td>7 </td><td>泸州临港产业管理有限公司 </td><td>10,000.00 </td><td>100.00 </td><td>二级 </td><td>服务 </td></tr><tr><td>8 </td><td>泸州港投汽车服务有限公司 </td><td>1,500.00 </td><td>100.00 </td><td>三级 </td><td>服务 </td></tr><tr><td>9 </td><td>泸州临港物业管理有限公司 </td><td>2,000.00 </td><td>100.00 </td><td>二级 </td><td>服务 </td></tr><tr><td>10 </td><td>泸州临港思源混凝土有限公司 </td><td>2,500.00 </td><td>51.00 </td><td>二级 </td><td>水泥制品 </td></tr><tr><td>11 </td><td>四川临港物流信息科技有限公司 </td><td>2,400.00 </td><td>95.00 </td><td>二级 </td><td>服务 </td></tr><tr><td>12 </td><td>泸州临港国际贸易有限公司 </td><td>6,800.00 </td><td>100.00 </td><td>二级 </td><td>贸易 </td></tr><tr><td>13 </td><td>泸州芯威科技有限公司 </td><td>10,000.00 </td><td>100.00 </td><td>三级 </td><td>服务 </td></tr><tr><td>14 </td><td>泸州临港自贸供应链管理有限公司 </td><td>100.00 </td><td>100.00 </td><td>三级 </td><td>贸易 </td></tr><tr><td>15 </td><td>泸州临港自贸(香港)有限公司 </td><td>50.00 </td><td>50.00 </td><td>三级 </td><td>贸易 </td></tr><tr><td>16 </td><td>泸州兴港物流有限公司 </td><td>5,000.00 </td><td>100.00 </td><td>二级 </td><td>服务 </td></tr><tr><td>17 </td><td>泸州港投智慧能源服务有限公司 </td><td>2,000.00 </td><td>60.00 </td><td>二级 </td><td>服务 </td></tr><tr><td>18 </td><td>泸州临港新型材料有限公司 </td><td>5,000.00 </td><td>100.00 </td><td>二级 </td><td>服务 </td></tr></table>"""
    tests = []
    tests.append(text)
    results = server_test(json.dumps(tests, ensure_ascii=False))
    print(results)
    # print(json.loads(results))
    # for i in tqdm(range(1000)):
    #    _ = server_test(json.dumps(test, ensure_ascii=False), "http://10.17.107.66:8019/prediction")

    # print(json.loads(aa))
