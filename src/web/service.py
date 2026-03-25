from json_repair.json_parser import JSONParser
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType

from configuration.config import *


class ChatService:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG["uri"],
            username=NEO4J_CONFIG["auth"][0],
            password=NEO4J_CONFIG["auth"][1],
        )
        # 嵌入模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name='BAAI/bge-base-zh-v1.5',
            encode_kwargs={'normalize_embeddings': True},
        )
        # LLM
        self.llm = ChatDeepSeek(model='deepseek-chat', api_key=API_KEY)
        # 定义所有实体对应的混合检索Neo4jVector对象
        self.neo4j_vectors = {
            'Trademark': Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name='trademark_vector_index',
                keyword_index_name='trademark_fulltext_index',
                search_type=SearchType.HYBRID,
            ),
            'SPU': Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name='spu_vector_index',
                keyword_index_name='spu_fulltext_index',
                search_type=SearchType.HYBRID,
            ),
            'SKU': Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name='sku_vector_index',
                keyword_index_name='sku_fulltext_index',
                search_type=SearchType.HYBRID,
            ),
            'Category1': Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name='category1_vector_index',
                keyword_index_name='category1_fulltext_index',
                search_type=SearchType.HYBRID,
            ),
            'Category2': Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name='category2_vector_index',
                keyword_index_name='category2_fulltext_index',
                search_type=SearchType.HYBRID,
            ),
            'Category3': Neo4jVector.from_existing_index(
                self.embedding_model,
                url=NEO4J_CONFIG["uri"],
                username=NEO4J_CONFIG["auth"][0],
                password=NEO4J_CONFIG["auth"][1],
                index_name='category3_vector_index',
                keyword_index_name='category3_fulltext_index',
                search_type=SearchType.HYBRID,
            )
        }
        # 定义Parser
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()

    # 核心聊天服务流程
    def chat(self, question):
        # 1. 根据用户问题，生成 Cypher以及需要对齐的实体
        result = self._generate_cypher(question)
        cypher = result['cypher_query']
        entities_to_align = result['entities_to_align']
        print(cypher)
        print("对齐之前的实体名称: ", entities_to_align)

        # 2. 实体对齐（混合检索）
        aligned_entities = self._entity_align(entities_to_align)
        print("对齐之后的实体名称: ", aligned_entities)

        # 3. 执行Cypher语句，得到查询结果
        query_result = self._execute_cypher(cypher, aligned_entities)
        print("查询结果: ", query_result)

        # 4. 根据用户问题和查询结果生成答案
        answer = self._generate_answer(question, query_result)
        print("最终回答: ", answer)
        return answer

    # 1. 根据问题，调用LLM生成Cypher
    def _generate_cypher(self, question):
        # 提示词
        prompt = """
                你是一个专业的Neo4j Cypher查询生成器。你的任务是根据用户问题生成一条Cypher查询语句，用于从知识图谱中获取回答用户问题所需的信息。

                用户问题：{question}

                知识图谱结构信息：{schema_info}

                要求：
                1. 生成参数化Cypher查询语句，用param_0, param_1等代替具体值
                2. 识别需要对齐的实体
                3. 必须严格使用以下JSON格式输出结果
                {{
                  "cypher_query": "生成的Cypher语句",
                  "entities_to_align": [
                    {{
                      "param_name": "param_0",
                      "entity": "原始实体名称",
                      "label": "节点类型"
                    }}
                  ]
                }}"""
        prompt = PromptTemplate.from_template(prompt)
        prompt = prompt.format(question=question, schema_info=self.graph.schema)
        # 得到模型输出
        output = self.llm.invoke(prompt)
        # 解析成JSON
        result = self.json_parser.invoke(output)
        return result

    # 2. 实体对齐（混合检索）
    def _entity_align(self, entities_to_align):
        # 遍历所有的实体
        for index, entity_to_align in enumerate(entities_to_align):
            label = entity_to_align['label']
            entity = entity_to_align['entity']
            # 混合检索，得到对齐后的实体名称
            aligned_entity = self.neo4j_vectors[label].similarity_search(entity, k=1)[0].page_content
            # 覆盖原来的实体名称
            entities_to_align[index]['entity'] = aligned_entity
        return entities_to_align

    # 3. 用对齐的实体名称替换param_0，执行Cypher
    def _execute_cypher(self, cypher, aligned_entities):
        # 提取对齐后的实体名称
        params = {aligned_entity['param_name'] : aligned_entity['entity'] for aligned_entity in aligned_entities}
        return self.graph.query(cypher, params=params)

    # 4. 生成回答
    def _generate_answer(self, question, query_result):
        prompt = """
                你是一个电商智能客服，根据用户问题，以及数据库查询结果生成一段简洁、准确的自然语言回答。
                用户问题: {question}
                数据库返回结果: {query_result}
        """
        prompt = prompt.format(question=question, query_result=query_result)
        output = self.llm.invoke(prompt)
        result = self.str_parser.invoke(output)
        return result

if __name__ == '__main__':
    chat_service = ChatService()
    chat_service.chat("Apple都有哪些产品？")