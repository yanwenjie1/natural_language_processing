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


def server_test(sen, url):
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
    text = '2022年崇仁县城市建设投资发展有限公司公司债券（品种一）交易场所选择应急申请书。'
    text = '二、本期债务融资工具主要条款：1、债务融资工具全称：三一集团有限公司2022年度第十四期@@超短期融资券（科创票据）@@2、计划发行规模：人民币壹拾亿元@@3、期限：65天@@4、债券面值：人民币100元@@5、发行日：2022年10月17日@@6、分销日：2022年10月18日@@7、缴款日：2022年10月18日@@8、上市日：2022年10月19日@@9、兑付日：2022年12月22日（如遇法定节假日或休息日，则顺延至其后的第1个工作日，顺延期间不另计息）。@@10、付息日：2022年12月22日（如遇法定节假日或休息日，则顺延至其后的第1个工作日，顺延期间不另计息）。@@11、本期债务融资工具的销售佣金费率(%/年化)不低于0%。@@12、债券评级：中诚信国际信用评级有限责任公司给予发行方的主体评级为AAA，债项评级为\。@@13、登记托管形式：本期债务融资工具采用实名制记账式，在银行间市场清算所股份有限公司（以下简称“上海清算所”）统一托管。@@14、付息及兑付方式：本期债务融资工具的利息支付及到期兑付事宜按托管机构的有关规定执行，由登记托管机构代理完成利息支付及本金兑付工作。@@'
    text = '原告：冯光森。,委托诉讼代理人：李俊瑶，山东滨蓝律师事务所律师。,被告：邢新卫。,委托诉讼代理人：张丽君，山东君晴律师事务所律师。,委托诉讼代理人：张督督，山东君晴律师事务所律师。,原告冯光森与被告邢新卫房屋租赁合同纠纷一案，本院受理后，依法适用简易程序，公开开庭进行了审理。原、被告的委托诉讼代理人均到庭参加了诉讼。约定被告将位于潍坊高新区永惠路117号宏臻－东方xx号商住楼xx号xx街商铺的房屋出租给原告作为餐饮使用。却故意隐瞒政府规定该房不能从事餐饮活动的事实。原告：冯光森。,委托诉讼代理人：李俊瑶，山东滨蓝律师事务所律师。,被告：邢新卫。,委托诉讼代理人：张丽君，山东君晴律师事务所律师。,委托诉讼代理人：张督督，山东君晴律师事务所律师。,原告冯光森与被告邢新卫房屋租赁合同纠纷一案，本院受理后，依法适用简易程序，公开开庭进行了审理。原、被告的委托诉讼代理人均到庭参加了诉讼。约定被告将位于潍坊高新区永惠路117号宏臻－东方xx号商住楼xx号xx街商铺的房屋出租给原告作为餐饮使用。却故意隐瞒政府规定该房不能从事餐饮活动的事实。原告：冯光森。,委托诉讼代理人：李俊瑶，山东滨蓝律师事务所律师。'
    texts = []
    for i in range(1):
        texts.append(text)
    results = server_test(json.dumps(texts, ensure_ascii=False), "http://10.17.107.66:12000/prediction")
    print(results)
    # results = json.loads(results)
    # for item in  results:
    #     print(item)
    for i in tqdm(range(10000)):
        results1 = server_test(json.dumps(texts, ensure_ascii=False), "http://127.0.0.1:12000/prediction")
        if results1 != results:
            print('wrong')
