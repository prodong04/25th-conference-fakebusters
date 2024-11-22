package com.fakebusters.dbust_backend

import io.github.cdimascio.dotenv.Dotenv
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class DbustBackendApplication

fun main(args: Array<String>) {
	// Load environment variables from .env file
	val dotenv = Dotenv.configure().load()
	System.setProperty("AWS_ACCESS_KEY", dotenv["AWS_ACCESS_KEY"])
	System.setProperty("AWS_SECRET_KEY", dotenv["AWS_SECRET_KEY"])

	runApplication<DbustBackendApplication>(*args)
}
