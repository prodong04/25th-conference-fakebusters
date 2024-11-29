package com.fakebusters.dbust_backend.controller

import com.fakebusters.dbust_backend.service.S3Service
import kotlinx.coroutines.runBlocking
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.*
import org.springframework.web.multipart.MultipartFile

@RestController
@RequestMapping("/api/files")
@CrossOrigin(origins = ["http://localhost:3000"])
class FileUploadController(private val s3Service: S3Service) {

    @PostMapping("/upload")
    fun uploadFile(@RequestParam("file") file: MultipartFile): ResponseEntity<FileUploadResponse> = runBlocking {
        val startTime = System.currentTimeMillis() // Start time
        println("Upload started at: ${java.time.Instant.now()}")

        val fileKey = s3Service.uploadFile(file)

        val endTime = System.currentTimeMillis() // End time
        val duration = endTime - startTime // Duration in milliseconds
        println("Upload completed in $duration milliseconds")


        return@runBlocking ResponseEntity.ok(FileUploadResponse(fileKey))
    }
}

data class FileUploadResponse(
    val fileKey: String
)