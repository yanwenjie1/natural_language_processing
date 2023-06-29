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
    url = 'http://10.17.107.66:12005/prediction'
    text = '3、本基金投资的前十名证券之一中信证券的发行主体中信证券股份有限公司于2023年2月6日因未按规定履行客户身份识别义务等，受到中国人民银行处罚（银罚决字〔2023〕6号）。' * 10
    tests = []
    tests.append(text)
    print(text)
    results = server_test(json.dumps(tests, ensure_ascii=False))
    # results = server_test(json.dumps(text, ensure_ascii=False))
    print(results)
    # print(json.loads(results))
    # for i in json.loads(results):
    #     print(i)
    for i in tqdm(range(1000)):
       _ = server_test(json.dumps(tests, ensure_ascii=False))
       # _ = server_test(json.dumps(test, ensure_ascii=False), "http://10.17.107.66:8019/prediction")

    # print(json.loads(aa))
