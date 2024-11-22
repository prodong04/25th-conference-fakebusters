package com.fakebusters.dbust_backend.service

import aws.sdk.kotlin.runtime.auth.credentials.StaticCredentialsProvider
import aws.sdk.kotlin.services.s3.S3Client
import aws.sdk.kotlin.services.s3.model.PutObjectRequest
import aws.smithy.kotlin.runtime.content.ByteStream
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import org.springframework.web.multipart.MultipartFile
import java.util.UUID

@Service
class S3Service {
    @Value("\${aws.s3.bucket}")
    private lateinit var bucketName: String

    @Value("\${aws.s3.region}")
    private lateinit var region: String

    @Value("\${AWS_ACCESS_KEY}")
    private lateinit var accessKey: String

    @Value("\${AWS_SECRET_KEY}")
    private lateinit var secretKey: String

    suspend fun uploadFile(file: MultipartFile): String {
        val fileKey = generateUniqueFileName(file.originalFilename)

        S3Client {
            region = this@S3Service.region
            credentialsProvider = StaticCredentialsProvider {
                accessKeyId = this@S3Service.accessKey
                secretAccessKey = this@S3Service.secretKey
            }
        }.use { s3Client ->
            val request = PutObjectRequest {
                bucket = bucketName
                key = fileKey
                body = ByteStream.fromBytes(file.bytes)
                contentType = file.contentType
            }

            s3Client.putObject(request)
        }

        return fileKey
    }

    private fun generateUniqueFileName(originalFileName: String?): String {
        val extension = originalFileName?.substringAfterLast('.', "")
        return "${UUID.randomUUID()}.$extension"
    }
}