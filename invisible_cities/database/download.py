import sys
import pymysql
pymysql.install_as_MySQLdb()
import os
import re
from os import path

# Absolute imports to allow usage as standalone program:
# python invisible_cities/database/download.py
from invisible_cities.database.db_connection import connect_sqlite
from invisible_cities.database.db_connection import connect_mysql


def create_table_sqlite(cursorSqlite, cursorMySql, table):
    """Create a SQLite table from a MySQL table definition.

    Fetches the CREATE TABLE statement from MySQL, strips MySQL-specific
    syntax, and executes it on SQLite.

    Parameters
    ----------
    cursorSqlite : sqlite3.Cursor
        SQLite cursor.
    cursorMySql : pymysql.cursors.Cursor
        MySQL cursor.
    table : str
        Name of the table to copy.
    """
    cursorMySql.execute('show create table {}'.format(table))
    data = cursorMySql.fetchone()
    sql  = data[1]

    sql = re.sub(r" COMMENT\s+\'.*\'", "", sql)
    sql = re.sub(r"\s*ENGINE.*", "", sql)
    sql = re.sub(r",\s*\n\s*KEY.*[\n,]", "", sql)

    cursorSqlite.execute(sql)


def copy_all_rows(connSqlite, cursorSqlite, cursorMySql, table):
    """Copy all rows from a MySQL table to a SQLite table.

    Parameters
    ----------
    connSqlite : sqlite3.Connection
        SQLite connection.
    cursorSqlite : sqlite3.Cursor
        SQLite cursor.
    cursorMySql : pymysql.cursors.Cursor
        MySQL cursor.
    table : str
        Name of the table to copy.
    """
    cursorMySql.execute('SELECT * from {0}'.format(table))
    data = cursorMySql.fetchall()

    fields = '?'
    try:
        nfields = len(data[0])
        fields += (nfields-1) * ',?'
        cursorSqlite.executemany('INSERT INTO {0} VALUES({1})'.format(table,fields),data)
        connSqlite.commit()
    except IndexError:
        print('Table ' +table+' is empty.')


def loadDB(dbname : str, tables : list):
    """Clone MySQL database tables to a local SQLite database.

    Parameters
    ----------
    dbname : str
        Name of the MySQL database.
    tables : list of str
        List of table names to copy.
    """
    print("Cloning database {}".format(dbname))
    dbfile = path.join(os.environ['ICDIR'], 'database/localdb.'+dbname+'.sqlite3')
    try:
        os.remove(dbfile)
    except:
        pass

    conn_sqlite, cursor_sqlite = connect_sqlite(dbfile)
    conn_mysql , cursor_mysql  = connect_mysql (dbname)

    for table in tables:
        print("Downloading table {}".format(table))
        create_table_sqlite(cursor_sqlite, cursor_mysql, table)
        copy_all_rows(conn_sqlite, cursor_sqlite, cursor_mysql, table)


dbnames        = ('NEWDB', 'DEMOPPDB', 'NEXT100DB', 'Flex100DB')
common_tables  = ('DetectorGeo','PmtBlr','ChannelGain','ChannelMapping','ChannelMask',
                  'PmtNoiseRms','ChannelPosition','SipmBaseline', 'SipmNoisePDF',
                  'PMTFEMapping', 'PMTFELowFrequencyNoise')
extended       = dict(NEXT100DB = ("Activity", "Efficiency"))

table_dict = dict.fromkeys(dbnames, common_tables)
for dbname, extra in extended.items():
    table_dict[dbname] += extra

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dbname = sys.argv[1]
        loadDB(dbname, table_dict[dbname])
    else:
        for dbname, tables in table_dict.items():
            loadDB(dbname, tables)
