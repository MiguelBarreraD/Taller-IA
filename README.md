# Taller-IA

## Itorducción

Este taller explora cómo utilizar LangChain, Pinecone y OpenAI para enviar solicitudes a ChatGPT y obtener respuestas. LangChain es una biblioteca que proporciona una abstracción sobre la integración con modelos de lenguaje, lo que facilita el envío de solicitudes y la obtención de respuestas. Pinecone es un servicio de almacenamiento de vectores que se puede utilizar para construir sistemas de recuperación de respuestas. OpenAI es una empresa de inteligencia artificial que proporciona modelos de lenguaje avanzados, como ChatGPT.

## Requisitos

* jupyterlab
* openai
* tiktoken
* langchain
* openai
* chromadb
* langchainhub
* bs4
* pinecone-client
* langchain-pinecone
* langchain-community

## Instalación

Puede instalar las dependencias necesarias ejecutando el siguiente comando:

```bash
    pip install -r requirements.txt
```

> [!IMPORTANT]
> Para que cada uno de los ejercicios funcione correctamente, debe utilizar una OPENAI_API_KEY válida.

## Uso de LangChain, Pinecone y OpenAI para enviar solicitudes a ChatGPT y obtener respuestas:


### Ejecución

```bash
    py IA.py
```

### Descripción

1. Manipulamos las variables de entorno y establecemos la clave de la API de OpenAI como una variable de entorno.

```python
    template = """Question: {question}
    Answer: Let's think step by step."""
```

2. Definimos una plantilla que contiene un marcador de posición para la pregunta.

```python
    prompt = PromptTemplate(template=template, input_variables=["question"])
```

3. Creamos un objeto PromptTemplate que tomará la plantilla y una lista de variables de entrada, en este caso, solo "question".

```python
    llm = OpenAI()
```

4. Inicializamos un objeto OpenAI, presumiblemente una clase que encapsula la lógica de comunicación con la API de OpenAI.

```python
    llm_chain = LLMChain(prompt=prompt, llm=llm)
```
5. Creamos un objeto LLMChain (supongo que significa Language Model Chain) que toma el objeto PromptTemplate y el objeto OpenAI.

```python
    question = "What is at the core of Popper's theory of science?"
    response = llm_chain.run(question)
```

6. Definimos una pregunta y la pasamos a través de la cadena de modelos de lenguaje con el método run().

```python
    print(response)
```
### Prueba

#### Entrada

 **¿What is at the core of Popper's theory of science?**

#### Salida

## Creación de un sistema de recuperación de respuestas con almacenamiento de vectores en memoria (RAG):

### Ejecución

```bash
    py memoryvector.py
```
### Descripción


1. Configuramos un cargador de documentos web para extraer contenido de una página específica y cargamos los documentos.
```python
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
```

2. Dividimos los documentos en fragmentos de texto utilizando un método de división específico y luego imprimimos los primeros fragmentos.
```python
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(splits[0])
    print(splits[1])
```

3. Creamos un vectorstore (almacén de vectores) a partir de los fragmentos de texto divididos y convertimos este vectorstore en un "retriever" (recuperador) para buscar documentos relevantes.
```python
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
```

4. Cargamos un prompt RAG (Retrieve and Generate) y configuramos un modelo de lenguaje de ChatOpenAI.
```python
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
```

5. Definimos una cadena de procesamiento que utiliza el retriever para proporcionar contexto, pasa la pregunta sin cambios, aplica el prompt RAG y finalmente analiza la salida en formato de cadena.
```python
    def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
```

6. Invocamos la cadena de procesamiento con una pregunta específica y mostramos la respuesta generada.
```python
    response = rag_chain.invoke("What is Task Decomposition?")
    print(response)
```

### Prueba

#### Entrada

**¿What is Task Decomposition?**

#### Salida

## Creación de un sistema de recuperación de respuestas (RAG) utilizando Pinecone:

### Ejecución

```bash
    py RAG.py
```

### Descripción

#### Configuración y creación de índice en Pinecone:

Cargamos un archivo de texto y lo dividimos en documentos utilizando un procesador específico.

```python
    def loadText():
        loader = TextLoader("Conocimiento.txt")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
            is_separator_regex = False,
        )

        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
```

#### Búsqueda de documentos similares:

Configuramos Pinecone y creamos un nuevo índice si no existe uno ya creado.

```python
        import pinecone

    index_name = "langchain-demo"
    pc = Pinecone(api_key='c25ab479-3511-494a-b968-02ed72cf92a0')

    print(pc.list_indexes())

    if len(pc.list_indexes())==0:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(
                environment=os.getenv("PINECONE_ENV"),
                pod_type="p1.x1",
                pods=1
            )
        )
```
#### Búsqueda de documentos similares:

Realizamos una búsqueda de documentos similares utilizando Pinecone y mostramos el contenido del documento más relevante.

```python
    def search():
    embeddings = OpenAIEmbeddings()

    index_name = "langchain-demo"
    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

    query = "What is the principal idea text"
    docs = docsearch.similarity_search(query)

    print(docs[0].page_content)
```

#### Llamadas a las funciones:

Finalmente, llamamos a las funciones para ejecutar el proceso de carga de texto y la búsqueda de documentos similares.
```python
    loadText()
    search()
```


### Prueba

#### Entrada
**¿What is the principal idea text?**
#### Salida

