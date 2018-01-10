import pandas as pd
import operator
import pymysql
import time
import json
import datetime
import requests
import csv
import re
import numpy as np
from datetime import date


def load_csv(fname):
    csv = pd.read_csv(fname)
    return csv

def getCategory(uri):
    """ return: empty str '' => failed to get category
        PS: "Unclassified" is valid (returned by FortiGuard)
    """
    API = 'http://www.fortiguard.com/webfilter?q='
    cat = ''
    try:
        # import requests
        r = requests.get(API + uri, timeout=60)
        # print r.content[:200]
        # print len(r.content)
        html = r.content

        # import re
        result = re.search(b'Category:(.*)" />', html)  # previous version
        if result is None:  # 2nd chance
            result = re.search(b':(.*)</h4>', html)  # since 2015-08-31

        cat = result.group(1).strip()

    except:  # more exceptions & errors -- ref: http://docs.python-requests.org/en/latest/user/quickstart/
        #print "getCategory error: " + uri
        import traceback
        traceback.print_exc()
        # f = open('e:\\tmp\\error.html', 'w+')
        # f.write(uri + ":\n" + html)
        # f.close()


    ## based on urllib2:
    ## Q: why not urllib2? (http://stackoverflow.com/questions/2018026/should-i-use-urllib-or-urllib2-or-requests)
    #     try:
    #         # import urllib2
    #         response = urllib2.urlopen(API+uri)
    #         html = response.read()  # print html
    #         # print html[:200]
    #         print len(html)
    #
    #         # import re
    #         result = re.search('Category:(.*)</h3> <a', html)
    #         cat = result.group(1).strip()
    #     except AttributeError:
    #         print "Invalid Response: " + uri
    #         print html[:1000]
    # #            f = open('e:\\tmp\\error.html', 'w+')
    # #            f.write(uri+":\n" + html)
    # #            f.close()
    #     except httplib.BadStatusLine:
    #         print "BadStatusLine(httplib): " + uri
    #     except: # e.g. URLError
    #         print "more getCategory error: " + uri
    #         import traceback
    #         traceback.print_exc()

    # if cat is None or cat == 'Unclassified':
    if cat is None:
        cat = ''

    # cat = 'TEST'

    #print "getCategory: " + uri + ":" + cat
    if len(cat) > 50:
        # raise ValueError
        print("getCategory warning: len(cat) > 50 -- " + cat)
        # cat = ''
    return cat

def load_cate_list():
    data = pd.read_csv('web_cate_DB.csv')
    __cate__ = {}
    for i in range(0, len(data), 1):
        __cate__[data['url'][i]] = data['cate'][i]
    return __cate__

def main():
    count = 1
    print("loading csv")
    __cate__ = load_cate_list()
    print("loading complete")

    for i in range(1, 8, 1):
        data_out = []
        fname = "all_user_csv_out_" + str(i) + ".csv"
        data = load_csv(fname)
        for j in range(len(data)):
            tmp_dict = {}
            tmp_dict.update({"id":data["id"][j]})
            tmp_dict.update({"visit_time":data["visit_time"][j]})


            if data["domain"][j] == "www.facebook.com":
                tmp_dict.update({'cate':"www.facebook.com"})
                #print("FB")
                data_out.append(tmp_dict)

            elif data["domain"][j] == "www.google.com.tw":
                tmp_dict.update({'cate':"www.google.com.tw"})
                #print("GOOGLE")
                data_out.append(tmp_dict)

            elif data["domain"][j] == "mail.google.com":
                tmp_dict.update({'cate':"mail.google.com"})
                #print("GMAIL")
                data_out.append(tmp_dict)

            elif data["domain"][j] == "www.youtube.com":
                tmp_dict.update({'cate':"www.youtube.com"})
                #print("YOUTUBE")
                data_out.append(tmp_dict)

            elif data["domain"][j] in __cate__:
                tmp_dict.update({'cate':__cate__[data["domain"][j]]})
                data_out.append(tmp_dict)

            else :
                print("strange domain")
                print("file: "+str(i) + " " +str(j) + "/" + str(len(data)))
                tmp_cate = getCategory(data["domain"][j])
                tmp_cate = str(tmp_cate)
                tmp_dict.update({'cate':tmp_cate})
                data_out.append(tmp_dict)
                __cate__[data["domain"][j]] = tmp_cate

        with open('all_user_csv_out_v2_' + str(i) + '.csv','w') as f:
            wr = csv.writer(f)
            tmp_title = ["id", "visit_time", "cate"]
            wr.writerow(tmp_title)
            for i in range(len(data_out)):
                values = []
                values.append(str(data_out[i]['id']))
                values.append(data_out[i]['visit_time'])
                values.append(data_out[i]['cate'])
                wr.writerow(values)







if __name__ == '__main__':
    main()
