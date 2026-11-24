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

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class Main {

    private static final Path VECTOR_STORE_PATH = Paths.get("vector_store.json");
    private static final Path DOCUMENTS_PATH = Paths.get("documents");

    interface Assistant {
        String chat(String userMessage);
    }

    public static void main(String[] args) {
        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("HATA: GEMINI_API_KEY ortam değişkeni bulunamadı!");
            return;
        }

        // 1. Gemini LLM Modelini Başlat
        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .timeout(java.time.Duration.ofSeconds(30))
                .maxRetries(1)
                .build();

        // 2. İngilizce Lokal Embedding Modeli
        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        // 3. Kalıcı Vektör Deposunu Yükle veya İlk Kez Oluştur
        InMemoryEmbeddingStore<TextSegment> embeddingStore;

        if (Files.exists(VECTOR_STORE_PATH)) {
            System.out.println("Mevcut vektör veritabanı diskten yükleniyor...");
            embeddingStore = InMemoryEmbeddingStore.fromFile(VECTOR_STORE_PATH);
        } else {
            System.out.println("Yeni vektör veritabanı oluşturuluyor...");
            embeddingStore = new InMemoryEmbeddingStore<>();

            List<Document> documents = FileSystemDocumentLoader.loadDocuments(DOCUMENTS_PATH);
            System.out.println(documents.size() + " adet ders notu okundu.");

            EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                    .documentSplitter(DocumentSplitters.recursive(500, 100))
                    .embeddingModel(embeddingModel)
                    .embeddingStore(embeddingStore)
                    .build();

            ingestor.ingest(documents);

            embeddingStore.serializeToFile(VECTOR_STORE_PATH);
            System.out.println("Vektörler diske kaydedildi: " + VECTOR_STORE_PATH.toAbsolutePath());
        }

        // 4. Retrieval Ayarları
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(4)
                .minScore(0.6)
                .build();

        // 5. RAG Asistanı
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