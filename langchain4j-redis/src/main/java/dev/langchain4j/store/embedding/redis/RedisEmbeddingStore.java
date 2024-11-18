package dev.langchain4j.store.embedding.redis;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.*;
import redis.clients.jedis.json.Path2;
import redis.clients.jedis.search.*;

import java.util.*;

import static dev.langchain4j.internal.Utils.*;
import static dev.langchain4j.internal.ValidationUtils.*;
import static dev.langchain4j.store.embedding.redis.RedisSchema.SCORE_FIELD_NAME;
import static java.lang.String.format;
import static java.util.Arrays.asList;
import static java.util.Collections.singletonList;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;
import static redis.clients.jedis.search.IndexDefinition.Type.JSON;
import static redis.clients.jedis.search.RediSearchUtil.ToByteArray;

/**
 * Represents a <a href="https://redis.io/">Redis</a> index as an embedding store.
 * Current implementation assumes the index uses the cosine distance metric.
 */
public class RedisEmbeddingStore implements EmbeddingStore<TextSegment> {

    private static final Logger log = LoggerFactory.getLogger(RedisEmbeddingStore.class);
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private JedisPooled pooledClient = null;
    private JedisCluster clusterClient = null;
    private final RedisSchema schema;

    /**
     * Creates an instance of RedisEmbeddingStore
     *
     * @param host         Redis Stack Server host
     * @param port         Redis Stack Server port
     * @param user         Redis Stack username (optional)
     * @param password     Redis Stack password (optional)
     * @param indexName    The name of the index (optional). Default value: "embedding-index".
     * @param dimension    Embedding vector dimension
     * @param metadataKeys Metadata keys that should be persisted (optional)
     */
    public RedisEmbeddingStore(String host,
                               Integer port,
                               String user,
                               String password,
                               String indexName,
                               Integer dimension,
                               Collection<String> metadataKeys,
                               Boolean isCluster
    ) {
        ensureNotBlank(host, "host");
        ensureNotNull(port, "port");
        ensureNotNull(dimension, "dimension");

        try {
            log.info("Creating RedisEmbeddingStore with host={}, port={}, user={}, indexName={}, dimension={}, metadataKeys={}",
                    host, port, user, indexName, dimension, metadataKeys);

            if (isCluster == null || !isCluster) {
                this.pooledClient = user == null ? new JedisPooled(host, port) : new JedisPooled(host, port, user, password);
            } else {
                JedisClientConfig jedisClientConfig = DefaultJedisClientConfig.builder().clientName("langchain4j")
                        .timeoutMillis(20_000).connectionTimeoutMillis(30_000)
                        .blockingSocketTimeoutMillis(0)
                        .socketTimeoutMillis(60_000)
                        .ssl(true).user(user).password(password).build();
                Set<HostAndPort> hostAndPortSet = new HashSet<>();
                hostAndPortSet.add(new HostAndPort(host, port));

                this.clusterClient = new JedisCluster(hostAndPortSet, jedisClientConfig, 2);
            }

            this.schema = RedisSchema.builder()
                    .indexName(getOrDefault(indexName, "embedding-index"))
                    .dimension(dimension)
                    .metadataKeys(metadataKeys)
                    .build();

            if (!isIndexExist(schema.indexName())) {
                createIndex(schema);
            }
        } catch (Exception e) {
            throw new RedisRequestFailedException("failed to create RedisEmbeddingStore", e);
        }
    }

    @Override
    public String add(Embedding embedding) {
        String id = randomUUID();
        add(id, embedding);
        return id;
    }

    @Override
    public void add(String id, Embedding embedding) {
        addInternal(id, embedding, null);
    }

    @Override
    public String add(Embedding embedding, TextSegment textSegment) {
        String id = randomUUID();
        addInternal(id, embedding, textSegment);
        return id;
    }

    @Override
    public List<String> addAll(List<Embedding> embeddings) {
        List<String> ids = embeddings.stream()
                .map(ignored -> randomUUID())
                .collect(toList());
        addAllInternal(ids, embeddings, null);
        return ids;
    }

    @Override
    public List<String> addAll(List<Embedding> embeddings, List<TextSegment> embedded) {
        List<String> ids = embeddings.stream()
                .map(ignored -> randomUUID())
                .collect(toList());
        addAllInternal(ids, embeddings, embedded);
        return ids;
    }

    @Override
    public EmbeddingSearchResult<TextSegment> search(EmbeddingSearchRequest request) {
        String queryTemplate = " =>[ KNN %d @%s $BLOB AS %s ]";
        List<String> returnFields = new ArrayList<>(schema.metadataKeys());
        returnFields.addAll(asList(schema.vectorFieldName(), schema.scalarFieldName(), SCORE_FIELD_NAME));

        log.info("returnFields={}", returnFields);

        String formattedQuery;

        if (request.filter() != null) {
            formattedQuery = parseFilterToRedisExpression(request.filter().toString()) +
                    format(queryTemplate, request.maxResults(), schema.vectorFieldName(), SCORE_FIELD_NAME);
        } else {
            formattedQuery = "*" + format(queryTemplate, request.maxResults(), schema.vectorFieldName(), SCORE_FIELD_NAME);
        }

        log.info("formattedQuery={}", formattedQuery);

        Query query = new Query(formattedQuery)
                .addParam("BLOB", ToByteArray(request.queryEmbedding().vector()))
                .returnFields(returnFields.toArray(new String[0]))
                .setSortBy(SCORE_FIELD_NAME, true)
                .dialect(2);

        SearchResult result = getSearchResult(query);
        List<Document> documents = result.getDocuments();
        log.info("search result document size: {}", documents.size());

        return new EmbeddingSearchResult<>(toEmbeddingMatch(documents, request.minScore()));
    }

    @Override
    public List<EmbeddingMatch<TextSegment>> findRelevant(Embedding referenceEmbedding, int maxResults, double minScore) {
        // Using KNN query on @vector field
        String queryTemplate = "*=>[ KNN %d @%s $BLOB AS %s ]";
        List<String> returnFields = new ArrayList<>(schema.metadataKeys());
        returnFields.addAll(asList(schema.vectorFieldName(), schema.scalarFieldName(), SCORE_FIELD_NAME));
        Query query = new Query(format(queryTemplate, maxResults, schema.vectorFieldName(), SCORE_FIELD_NAME))
                .addParam("BLOB", ToByteArray(referenceEmbedding.vector()))
                .returnFields(returnFields.toArray(new String[0]))
                .setSortBy(SCORE_FIELD_NAME, true)
                .dialect(2);

        SearchResult result = getSearchResult(query);

        List<Document> documents = result.getDocuments();

        return toEmbeddingMatch(documents, minScore);
    }

    private SearchResult getSearchResult(Query query) {
        SearchResult result;

        if (clusterClient != null) {
            log.info("Searching={} in index: {}", query, schema.indexName());
            result = clusterClient.ftSearch(schema.indexName(), query);
        } else {
            result = pooledClient.ftSearch(schema.indexName(), query);
        }
        return result;
    }

    private void createIndex(RedisSchema redisSchema) {
        String indexName = redisSchema.indexName();
        IndexDefinition indexDefinition = new IndexDefinition(JSON);
        indexDefinition.setPrefixes(formatPrefix(schema.prefix()));

        String res;
        if (clusterClient != null) {
            try {
                String formattedPrefix = formatPrefix(schema.prefix());
                log.debug("Creating index: {} with prefix: {}", indexName, formattedPrefix);
                res = clusterClient.ftCreate(indexName, FTCreateParams.createParams()
                        .on(IndexDataType.JSON).addPrefix(formattedPrefix), schema.toSchemaFields());
            } catch (Exception e) {
                log.warn("create index error, msg={}", e.getMessage());
                res = e.getMessage();
            }
        } else {
            res = pooledClient.ftCreate(indexName, FTCreateParams.createParams()
                    .on(IndexDataType.JSON)
                    .addPrefix(schema.prefix()), schema.toSchemaFields());
        }
        if (!"OK".equals(res) && !res.contains("broadcasting")) {
            if (log.isErrorEnabled()) {
                log.error("create index error, msg={}", res);
            }
            throw new RedisRequestFailedException("create index error, msg=" + res);
        }

    }

    private boolean isIndexExist(String indexName) {
        Set<String> indexes;
        if (clusterClient != null) {
            indexes = clusterClient.ftList();
        } else {
            indexes = pooledClient.ftList();
        }
        return indexes.contains(indexName);
    }

    private void addInternal(String id, Embedding embedding, TextSegment embedded) {
        addAllInternal(singletonList(id), singletonList(embedding), embedded == null ? null : singletonList(embedded));
    }

    private void addAllInternal(List<String> ids, List<Embedding> embeddings, List<TextSegment> embedded) {
        if (isNullOrEmpty(ids) || isNullOrEmpty(embeddings)) {
            log.info("do not add empty embeddings to redis");
            return;
        }
        ensureTrue(ids.size() == embeddings.size(), "ids size is not equal to embeddings size");
        ensureTrue(embedded == null || embeddings.size() == embedded.size(), "embeddings size is not equal to embedded size");

        List<Object> responses;
        if (clusterClient != null) {
            responses = getClusteredResponses(ids, embeddings, embedded);
        } else {
            responses = getPooledResponses(ids, embeddings, embedded);
        }

        Optional<Object> errResponse = responses.stream().filter(response -> !"OK".equals(response))
                .filter(response -> !response.toString().contains("Response String")).findAny();
        if (errResponse.isPresent()) {
            if (log.isErrorEnabled()) {
                log.error("add embedding failed, msg={}", errResponse.get());
            }
            throw new RedisRequestFailedException("add embedding failed, msg=" + errResponse.get());
        }
    }

    private List<Object> getClusteredResponses(List<String> ids, List<Embedding> embeddings, List<TextSegment> embedded) {
        List<Object> responses = new ArrayList<>();
        try (ClusterPipeline pipeline = clusterClient.pipelined()) {

            int size = ids.size();
            for (int i = 0; i < size; i++) {
                String id = ids.get(i);
                Embedding embedding = embeddings.get(i);
                TextSegment textSegment = embedded == null ? null : embedded.get(i);
                Map<String, Object> fields = new HashMap<>();
                fields.put(schema.vectorFieldName(), embedding.vector());
                if (textSegment != null) {
                    // do not check metadata key is included in RedisSchema#metadataKeys
                    fields.put(schema.scalarFieldName(), textSegment.text());
                    fields.putAll(textSegment.metadata().asMap());
                }
                String key = String.format("%s%s", formatPrefix(schema.prefix()), id);

                log.info("Saving key: {} with fields: {}", key, fields.keySet());
                responses.add(pipeline.jsonSetWithEscape(key, Path2.of("$"), fields));
            }

            pipeline.sync();
        }
        return responses;
    }

    String extractId(String input) {
        if (input.contains(":")) {
            String[] parts = input.split(":");
            return parts[parts.length - 1];
        }
        return input;
    }

    private String formatPrefix(String prefix) {
        if (clusterClient != null) {
            if (prefix.endsWith(":")) {
                String key = prefix.substring(0, prefix.length() - 1);
                return "{" + key + "}:";
            } else {
                return "{" + prefix + "}:";
            }
        } else {
            return prefix;
        }
    }

    private List<Object> getPooledResponses(List<String> ids, List<Embedding> embeddings, List<TextSegment> embedded) {
        List<Object> responses;
        try (Pipeline pipeline = pooledClient.pipelined()) {

            int size = ids.size();
            for (int i = 0; i < size; i++) {
                String id = ids.get(i);
                Embedding embedding = embeddings.get(i);
                TextSegment textSegment = embedded == null ? null : embedded.get(i);
                Map<String, Object> fields = new HashMap<>();
                fields.put(schema.vectorFieldName(), embedding.vector());
                if (textSegment != null) {
                    // do not check metadata key is included in RedisSchema#metadataKeys
                    fields.put(schema.scalarFieldName(), textSegment.text());
                    fields.putAll(textSegment.metadata().asMap());
                }
                String key = schema.prefix() + id;
                pipeline.jsonSetWithEscape(key, Path2.of("$"), fields);
            }

            responses = pipeline.syncAndReturnAll();
        }
        return responses;
    }

    private List<EmbeddingMatch<TextSegment>> toEmbeddingMatch(List<Document> documents, double minScore) {
        if (documents == null || documents.isEmpty()) {
            return new ArrayList<>();
        }

        return documents.stream()
                .map(document -> {
                    double score = (2 - Double.parseDouble(document.getString(SCORE_FIELD_NAME))) / 2;
                    String id = extractId(document.getId());
                    log.info("document={}", document);
                    log.info("document id={}", id);
                    log.info("scalarFieldName={}", schema.scalarFieldName());
                    String text = document.hasProperty(schema.scalarFieldName()) ? document.getString(schema.scalarFieldName()) : null;
                    TextSegment embedded = null;
                    if (text != null) {
                        Map<String, String> metadata = schema.metadataKeys().stream()
                                .filter(document::hasProperty)
                                .collect(toMap(metadataKey -> metadataKey, document::getString));
                        log.info("metadata={}", metadata);
                        embedded = new TextSegment(text, new Metadata(metadata));
                    }
                    Embedding embedding;
                    try {
                        log.info("vectorFieldName={}", schema.vectorFieldName());
                        log.info("parsed document={}", document.get(schema.vectorFieldName()));
                        document.getProperties().forEach((key) -> log.info("Property name: {}", key));

                        float[] vectors;
                        String vectorFieldName = schema.vectorFieldName();
                        if (!document.hasProperty(vectorFieldName)) {
                            log.warn("Vector field name: {} is null, trying MemoryDB format", vectorFieldName);
                            try (JsonParser parser = OBJECT_MAPPER.readTree(document.getString("$"))
                                    .get(0)
                                    .get("vector")
                                    .traverse(OBJECT_MAPPER)) {
                                vectors = parser.readValueAs(float[].class);
                            }
                        } else {
                            vectors = OBJECT_MAPPER.readValue(document.getString(vectorFieldName),
                                    float[].class);
                        }
                        embedding = new Embedding(vectors);
                    } catch (Exception e) {
                        throw new RedisRequestFailedException("failed to parse embedding", e);
                    }
                    return new EmbeddingMatch<>(score, id, embedding, embedded);
                })
                .filter(embeddingMatch -> embeddingMatch.score() >= minScore)
                .collect(toList());
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        private String host;
        private Integer port;
        private String user;
        private String password;
        private String indexName;
        private Integer dimension;
        private Collection<String> metadataKeys = new ArrayList<>();
        private Boolean isCluster = false;

        /**
         * @param host Redis Stack host
         */
        public Builder host(String host) {
            this.host = host;
            return this;
        }

        /**
         * @param port Redis Stack port
         */
        public Builder port(Integer port) {
            this.port = port;
            return this;
        }

        /**
         * @param user Redis Stack username (optional)
         */
        public Builder user(String user) {
            this.user = user;
            return this;
        }

        /**
         * @param password Redis Stack password (optional)
         */
        public Builder password(String password) {
            this.password = password;
            return this;
        }

        /**
         * @param indexName The name of the index (optional). Default value: "embedding-index".
         * @return builder
         */
        public Builder indexName(String indexName) {
            this.indexName = indexName;
            return this;
        }

        /**
         * @param dimension embedding vector dimension
         * @return builder
         */
        public Builder dimension(Integer dimension) {
            this.dimension = dimension;
            return this;
        }

        /**
         * @param metadataFieldsName metadata fields names (optional)
         * @deprecated use {@link #metadataKeys(Collection)} instead
         */
        @Deprecated
        public Builder metadataFieldsName(Collection<String> metadataFieldsName) {
            this.metadataKeys = metadataFieldsName;
            return this;
        }

        /**
         * @param metadataKeys Metadata keys that should be persisted (optional)
         */
        public Builder metadataKeys(Collection<String> metadataKeys) {
            this.metadataKeys = metadataKeys;
            return this;
        }

        /**
         * @param isCluster whether to use Redis Cluster
         */
        public Builder isCluster(Boolean isCluster) {
            this.isCluster = isCluster;
            return this;
        }

        public RedisEmbeddingStore build() {
            return new RedisEmbeddingStore(host, port, user, password, indexName, dimension, metadataKeys, isCluster);
        }
    }

    public String parseFilterToRedisExpression(String filterString) {

        if (!filterString.startsWith("IsEqualTo")) {
            throw new IllegalArgumentException("Unsupported filter operation");
        }

        int start = filterString.indexOf('(');
        int end = filterString.lastIndexOf(')');
        if (start == -1 || end == -1) {
            throw new IllegalArgumentException("Invalid filter format: missing parentheses");
        }

        // Get the key-value pairs
        String[] pairs = filterString.substring(start + 1, end).split(",");
        String fieldName = null;
        String value = null;

        for (String pair : pairs) {
            String[] keyValue = pair.trim().split("=");
            if (keyValue.length != 2) {
                throw new IllegalArgumentException("Invalid key-value pair: " + pair);
            }

            if (keyValue[0].equals("key")) {
                // Remove the "$." prefix from the field name
                fieldName = keyValue[1];
            } else if (keyValue[0].equals("comparisonValue")) {
                value = keyValue[1];
            }
        }

        if (fieldName == null || value == null) {
            throw new IllegalArgumentException("Missing required fields in filter");
        }

        return String.format("(@%s:{%s})", fieldName, value);
    }
}

