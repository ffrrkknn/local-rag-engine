package org.example;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Stream;

public class Main {

    private static final Path VECTOR_STORE_PATH = Paths.get("vector_store.json");
    private static final Path INDEXED_FILES_PATH = Paths.get("indexed_files.txt");
    private static final Path DOCUMENTS_PATH = Paths.get("documents");

    interface Assistant {
        String chat(String userMessage);
    }

    public static void main(String[] args) throws IOException {
        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("HATA: GEMINI_API_KEY ortam değişkeni bulunamadı!");
            return;
        }

        // 1. Gemini LLM Modelini Başlat
        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .timeout(Duration.ofSeconds(30))
                .maxRetries(1)
                .build();

        // 2. İngilizce Lokal Embedding Modeli
        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        // 3. Mevcut Vektör Deposunu Yükle veya Sıfırdan Oluştur
        InMemoryEmbeddingStore<TextSegment> embeddingStore = Files.exists(VECTOR_STORE_PATH)
                ? InMemoryEmbeddingStore.fromFile(VECTOR_STORE_PATH)
                : new InMemoryEmbeddingStore<>();

        // 4. Daha Önce İndekslenmiş Dosyaların Listesini Oku
        Set<String> indexedFiles = new HashSet<>();
        if (Files.exists(INDEXED_FILES_PATH) && Files.exists(VECTOR_STORE_PATH)) {
            indexedFiles.addAll(Files.readAllLines(INDEXED_FILES_PATH));
        }

        // 5. documents/ Klasöründeki Yeni Dosyaları Tespit Et (Incremental Ingestion)
        if (!Files.exists(DOCUMENTS_PATH)) {
            Files.createDirectories(DOCUMENTS_PATH);
        }

        List<Document> newDocuments = new ArrayList<>();
        List<String> newlyIndexedNames = new ArrayList<>();

        try (Stream<Path> paths = Files.walk(DOCUMENTS_PATH)) {
            List<Path> files = paths.filter(Files::isRegularFile).toList();
            for (Path file : files) {
                // Dosya adı ve son değiştirilme zamanını benzersiz imza olarak kullan
                String fileSignature = file.getFileName().toString() + "_" + Files.getLastModifiedTime(file).toMillis();
                if (!indexedFiles.contains(fileSignature)) {
                    System.out.println("Yeni/güncellenmiş dosya tespit edildi: " + file.getFileName());
                    newDocuments.add(FileSystemDocumentLoader.loadDocument(file));
                    newlyIndexedNames.add(fileSignature);
                }
            }
        }

        if (!newDocuments.isEmpty()) {
            System.out.println(newDocuments.size() + " adet yeni doküman vektörleştiriliyor...");
            EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                    .documentSplitter(DocumentSplitters.recursive(500, 100))
                    .embeddingModel(embeddingModel)
                    .embeddingStore(embeddingStore)
                    .build();

            ingestor.ingest(newDocuments);

            // Güncel vektör deposunu ve dosya kayıt listesini diske yaz
            embeddingStore.serializeToFile(VECTOR_STORE_PATH);
            Files.write(INDEXED_FILES_PATH, newlyIndexedNames,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            System.out.println("Vektör veritabanı güncellendi.");
        } else {
            System.out.println("Tüm dokümanlar güncel. Mevcut vektör veritabanı kullanılıyor.");
        }

        // 6. Retrieval Ayarları
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(4)
                .minScore(0.6)
                .build();

        // 7. RAG Asistanı
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(retriever)
                .build();

        System.out.println("\nRAG Sistemi Hazır! Sorunu sorabilirsin (Çıkmak için 'exit'):");

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("\nSen: ");
            String question = scanner.nextLine();
            if ("exit".equalsIgnoreCase(question.trim())) {
                break;
            }
            String answer = assistant.chat(question);
            System.out.println("AI: " + answer);
        }
        scanner.close();
    }
}