from pytest import mark

import sqlite3
import pymysql
pymysql.install_as_MySQLdb()


def connect_sqlite(dbfile):
    """Connect to a SQLite database file.

    Parameters
    ----------
    dbfile : str
        Path to the SQLite database file.

    Returns
    -------
    tuple
        ``(connection, cursor)`` for the SQLite database.
    """
    conn_sqlite   = sqlite3.connect(dbfile)
    cursor_sqlite = conn_sqlite.cursor()
    return conn_sqlite, cursor_sqlite


@mark.skip(reason='server timeouts cause too many spurious test failures')
def connect_mysql(dbname):
    """Connect to the NEXT MySQL database server.

    Parameters
    ----------
    dbname : str
        Name of the database on the server.

    Returns
    -------
    tuple
        ``(connection, cursor)`` for the MySQL database.
    """
    conn_mysql  = pymysql.connect(host="next.ific.uv.es",
                                   user='nextreader',passwd='readonly', db=dbname)
    cursor_mysql  = conn_mysql .cursor()
    return connect_mysql, cursor_mysql
