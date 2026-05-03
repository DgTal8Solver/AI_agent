import uuid
import time
import threading

import chromadb

class VectorDB:
    def __init__(self, persist_path: str = None, collection_name: str = "agent_memory"):
        """
        Инициализация и (опционально) загрузка сохранённой векторной базы.
        Если persist_path не указан, база будет работать только в оперативной памяти (временно).
        """
        # Блокировка для обеспечения потокобезопасности при конкурентных вызовах
        self._lock = threading.Lock()
        
        if persist_path:
            # Инициализация клиента с сохранением на диск
            self._client = chromadb.PersistentClient(path=persist_path)
        else:
            # Инициализация временного клиента (удобно для тестов)
            self._client = chromadb.EphemeralClient()
            
        # Создаем или получаем существующую коллекцию (аналог таблицы в SQL)
        # Внутри этой коллекции ChromaDB будет сама генерировать эмбеддинги
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(self, text: str, metadata: dict = None) -> str:
        """Добавляет текст с метаданными, возвращает уникальный ID."""
        with self._lock:
            doc_id = str(uuid.uuid4())
            
            # Подготовка метаданных: ChromaDB требует, чтобы значения были базовых типов
            safe_metadata = metadata.copy() if metadata else {}
            
            # Автоматически сохраняем timestamp (время в секундах)
            safe_metadata["timestamp"] = int(time.time())
            
            # Добавляем документ в коллекцию
            self._collection.add(
                ids = [doc_id],
                documents = [text],
                metadatas = [safe_metadata]
            )
            
            return doc_id

    def search(self, query: str, top_k: int = 5, filter: dict = None) -> list[dict]:
        """
        Выполняет семантический поиск по базе.
        Возвращает список словарей:
        [
            {
                "id": str,
                "text": str,
                "score": float,
                "metadata": dict
            },
            ...
        ]
        """
        with self._lock:
            # Запрашиваем похожие документы
            results = self._collection.query(
                query_texts = [query],
                n_results = top_k,
                where = filter if filter else None
            )
            
            # Если ничего не найдено (или база пуста)
            if not results["ids"] or not results["ids"][0]:
                return []
                
            formatted_results = []
            
            # ChromaDB может принимать список запросов, поэтому возвращает списки списков.
            # Берем индекс 0, так как у нас всегда ровно один текстовый запрос (query).
            ids = results["ids"][0]
            documents = results["documents"][0]
            
            # Защита от пустых значений метаданных
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
            
            for i in range(len(ids)):
                formatted_results.append({
                    "id": ids[i],
                    "text": documents[i],
                    "score": distances[i], # В ChromaDB это дистанция. Чем она МЕНЬШЕ, тем ближе тексты.
                    "metadata": metadatas[i] or {}
                })
                
            return formatted_results

    def delete_by_id(self, id: str) -> bool:
        """Удаляет запись по ID. Возвращает True, если запись была удалена."""
        with self._lock:
            # Проверяем, существует ли запись
            existing = self._collection.get(ids=[id])
            if not existing["ids"]:
                return False
                
            self._collection.delete(ids=[id])
            return True

    def delete_by_filter(self, filter: dict) -> int:
        """Удаляет все записи, чьи метаданные совпадают с filter. Возвращает количество удалённых."""
        with self._lock:
            # Чтобы узнать количество удаленных, сначала находим их
            to_delete = self._collection.get(where=filter)
            count = len(to_delete["ids"])
            
            if count > 0:
                self._collection.delete(where=filter)
                
            return count
