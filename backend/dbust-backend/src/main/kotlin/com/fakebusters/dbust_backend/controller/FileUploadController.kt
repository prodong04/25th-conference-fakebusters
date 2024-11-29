package com.fakebusters.dbust_backend.controller

import com.fakebusters.dbust_backend.service.S3Service
import com.fakebusters.dbust_backend.service.CSVService
import kotlinx.coroutines.runBlocking
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.*
import org.springframework.web.multipart.MultipartFile

@RestController
@RequestMapping("/api/files")
@CrossOrigin(origins = ["http://localhost:3000"])
class FileUploadController(private val s3Service: S3Service, private val csvService: CSVService) {

    @PostMapping("/upload")
    fun uploadFile(@RequestParam("file") file: MultipartFile): ResponseEntity<FileUploadResponse> = runBlocking {
        val startTime = System.currentTimeMillis() // Start time
        println("Upload started at: ${java.time.Instant.now()}")

        val fileKey = s3Service.uploadFile(file)

        val endTime = System.currentTimeMillis() // End time
        println("Upload finished at: ${java.time.Instant.now()}")
        println("Upload duration: ${endTime - startTime} ms")

        // Log metrics to CSV

        val fileName = file.originalFilename ?: "unknown"
        val fileType = file.contentType ?: "unknown"
        val fileSize = file.size
        val uploadTime = endTime - startTime

        csvService.logUploadMetrics(fileName, fileType, fileSize, uploadTime)

        return@runBlocking ResponseEntity.ok(FileUploadResponse(fileKey))
    }
}

data class FileUploadResponse(
    val fileKey: String
)