import sqlite3


class OSRS_DB():
    '''
    OSRS_DB object is used to connect and interact with the main sqlite db.
    '''
    def __init__(self, dbPath) -> None:
        self.dbName = dbPath
        self.conn = sqlite3.connect(self.dbName)
        print(f"Connecting to {self.dbName}...")

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
    
    def writeQuery(self, query:str) -> None:
        try:
            # Create cursor object for queries
            cursor = self.conn.cursor()

            # Query db and store result
            cursor.execute(query)

            # Commit changes
            self.conn.commit()

        except Exception as e:
            print(f"An error occured: {e}")

    def closeConnection(self):
        # Handle db connection termination
        self.conn.close()
        print(f"Closing connection to {self.dbName}...")

