import pymysql

conn = pymysql.connect(host='127.0.0.1', user='root', passwd="123456", db='booklib')
cur = conn.cursor()
cur.execute("SELECT * from book where score >=9.4")
for r in cur:
    print(r)
cur.close()
conn.close()
