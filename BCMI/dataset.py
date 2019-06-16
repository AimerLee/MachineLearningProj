from typing import List

from tinydb import TinyDB


class Resolver:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def resolve(self) -> List[dict]:
        raise NotImplementedError("Dataset Resolver must implement the resolve method")


class SEED(TinyDB):
    def __init__(self, db_name='SEED'):
        super().__init__(db_name + '.json')

    def index_dataset(self, resolver: Resolver, table_name=TinyDB.DEFAULT_TABLE):
        """此处使用Resolver的目的是为了同一个SEED类能够同时加载多个dataset，用于跨数据库的数据整合"""
        rows = resolver.resolve()
        table = self.table(table_name)
        table.insert_multiple(rows)

    def global_search(self, cond, source_flag=False):
        """Table 全局搜索函数"""
        search_result = []
        for table_name in self.tables():
            table_search_result = self.table(table_name).search(cond)
            if not source_flag:
                search_result += table_search_result
            else:
                search_result += list((item, table_name) for item in table_search_result)
        return search_result
