import sqlite3


class OSRS_DB():
    '''
    OSRS_DB object is used to connect and interact with the main sqlite db.
    '''
    def __init__(self, name) -> None:
        self.dbName = f"{name}.db"
        self.conn = sqlite3.connect(self.dbName)

    def readQuery(self, query: str):
        try:
            # Create cursor object for queries
            cursor = self.conn.cursor()

            # Query db and store result
            cursor.execute(query)

            result = cursor.fetchone()

            return result
        except Exception as e:
            print(f"An error occured: {e}")
    
    def writeQuery(self, query:str):
        try:
            # Create cursor object for queries
            cursor = self.conn.cursor()

            # Query db and store result
            cursor.execute(query)

            # Commit changes
            self.conn.commit()

        except Exception as e:
            print(f"An error occured: {e}")


