from sqlalchemy.sql.functions import current_timestamp
from teradataml.dataframe.dataframe import DataFrame
import logging

class FeatureStore:

    def __init__(self, conn, metadata_table: str):
        self.metadata_table = metadata_table
        self.conn = conn
        if self._check_metadata():
            logging.debug("Metadata tables exist, all good")
        else:
            logging.warning("Missing metadata tables, creating it")
            self.create_metadata()

    def create_metadata(self):
        try:
            self.conn.execute(f'CT {self.metadata_table} (ENTITY_NAME VARCHAR(100), FEATURE_NAME VARCHAR(128), TABLE_NAME VARCHAR(300), ENTITY_KEY_COLUMN VARCHAR(128), FEATURE_COLUMN VARCHAR(128), DATE_COLUMN VARCHAR(128)) UNIQUE PRIMARY INDEX (ENTITY_NAME,FEATURE_NAME)')
        except Exception:
            logging.error(f'Couldn\'t create {self.metadata_table} table')
        try:
            self.conn.execute(f'CT {self.metadata_table + "_log"} (USER_NAME VARCHAR(100), ENTITY_NAME VARCHAR(128), FEATURES VARCHAR(4000), REQUEST_TS TIMESTAMP, REQUEST_QUERY VARCHAR(64000)) PRIMARY INDEX (USER_NAME,REQUEST_TS)')
        except Exception:
            logging.error(f'Couldn\'t create {self.metadata_table + "_log"} table')


    def delete_metadata(self):
        try:
            self.conn.execute(f'DROP TABLE {self.metadata_table}')
        except Exception:
            logging.error(f'Couldn\'t drop {self.metadata_table} table')

    def register_feature_group(self, entity, table, entity_key_column, features: dict, date_column):
        self.conn.execute(f'INS {self.metadata_table} (?,?,?,?,?,?)',
            [[entity, f, table, entity_key_column, features[f], date_column] for f in features])

    def delete_features(self):
        self.conn.execute(f'DEL {self.metadata_table}')

    def get_featureset_df(self, entity, date, feature_names: list):
        query = self.get_featureset_sql(entity, date, feature_names)
        self.conn.execute(f'INS {self.metadata_table + "_log"} (user,?,?,current_timestamp,?)',
        entity, ','.join(feature_names), query)
        return DataFrame.from_query(query=query, index_label = entity)

    def get_featureset_sql(self, entity, date, feature_names: list):
        t = []
        for f in feature_names:
            r = self.get_feature_details(entity,f)
            t.append(f'(SEL {r["entity_key_column"]} AS {entity}, {r["feature_column"]} AS {f} FROM {r["table"]} WHERE {r["date_column"]} = \'{date}\')')
        if len(t) == 1:
            query = f'SEL F0.{entity}, F0.{feature_names[0]} FROM {t[0]} AS F0'
        else:
            f = t[0]
            s = f'SEL F0.{entity}, F0.{feature_names[0]}'
            for i in range(1, len(t)):
                f = self._join_generator(f, t[i], i - 1, entity)
                s = self._sel_generator(s, feature_names[i], i - 1)
            query = f'{s},\'{date}\' AS feature_date FROM {f}'
        return query

    def get_metadata_table(self):
        return self.metadata_table

    def list_entities(self):
        result = self.conn.execute(f'SEL DISTINCT entity_name FROM {self.metadata_table}').fetchall()
        return self._flatten_list(result)

    def list_features(self, entity):
        result = self.conn.execute(f'SEL feature_name FROM {self.metadata_table} WHERE entity_name = ?', [entity]).fetchall()
        return self._flatten_list(result)

    def get_feature_details(self, entity, feature):

        result = self.conn.execute(f'SEL * FROM {self.metadata_table} WHERE entity_name = ? AND feature_name = ?', [entity, feature]).fetchall()
        result = self._flatten_list(result)
        feature_dict = {'entity': result[0],
                        'feature': result[1],
                        'table': result[2],
                        'entity_key_column': result[3],
                        'feature_column': result[4],
                        'date_column': result[5]}
        return feature_dict

    def _check_metadata(self):
        return self._check_table(self.metadata_table)

    def _check_log(self):
        return self._check_table(self.metadata_table + '_log')

    def _check_table(self, table):
        try:
            self.conn.execute(f'SEL TOP 1 * FROM {table}')
        except Exception:
            return False
        else:
            return True
    
    @staticmethod
    def _flatten_list(input):
        return [item for l in input for item in l]

    @staticmethod
    def _join_generator(a:str, b:str, pos: int, column: str):
        if pos == 0:
            return f'{a} AS F{pos} JOIN {b} AS F{pos + 1} ON F{pos}.{column} = F{pos + 1}.{column}'
        else:
            return f'{a} JOIN {b} AS F{pos + 1} ON F{pos}.{column} = F{pos + 1}.{column}'

    @staticmethod
    def _sel_generator(a:str, b:str, pos: int):
        return f'{a}, F{pos + 1}.{b}'



