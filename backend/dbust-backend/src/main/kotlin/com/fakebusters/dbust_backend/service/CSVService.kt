package com.fakebusters.dbust_backend.service

import org.springframework.stereotype.Service
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter

@Service
class CSVService {

    private val csvFilePath = "upload_metrics.csv"

    init {
        // Create the CSV file and write the header if it doesn't exist
        val file = File(csvFilePath)
        if (!file.exists()) {
            PrintWriter(FileWriter(file, true)).use { writer ->
                writer.println("file_name,file_type,file_size(B),upload_time(ms)")
            }
        }
    }

    fun logUploadMetrics(fileName: String, fileType: String, fileSize: Long, uploadTime: Long) {
        val file = File(csvFilePath)
        PrintWriter(FileWriter(file, true)).use { writer ->
            writer.println("$fileName,$fileType,$fileSize,$uploadTime")
        }
    }
}