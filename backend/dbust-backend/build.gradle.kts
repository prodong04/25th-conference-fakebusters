plugins {
	kotlin("jvm") version "1.9.25"
	kotlin("plugin.spring") version "1.9.25"
	id("org.springframework.boot") version "3.3.5"
	id("io.spring.dependency-management") version "1.1.6"
}

group = "com.fakebusters"
version = "0.0.1-SNAPSHOT"

java {
	toolchain {
		languageVersion = JavaLanguageVersion.of(21)
	}
}

repositories {
	mavenCentral()
}

dependencies {
	implementation("aws.sdk.kotlin:s3:1.0.30")
	implementation("com.squareup.okhttp3:okhttp:5.0.0-alpha.11")

	implementation("org.springframework.boot:spring-boot-starter")
	implementation("org.jetbrains.kotlin:kotlin-reflect")
	testImplementation("org.springframework.boot:spring-boot-starter-test")
	testImplementation("org.jetbrains.kotlin:kotlin-test-junit5")
	testRuntimeOnly("org.junit.platform:junit-platform-launcher")

	implementation("org.springframework.boot:spring-boot-starter-web") // For REST endpoints
	implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3") // For coroutines support
	implementation("org.jetbrains.kotlinx:kotlinx-coroutines-reactor:1.7.3") // For reactive support
	implementation("com.fasterxml.jackson.module:jackson-module-kotlin") // For JSON handling

	implementation("io.github.cdimascio:dotenv-kotlin:6.2.2") // For environment variable handling
}

kotlin {
	compilerOptions {
		freeCompilerArgs.addAll("-Xjsr305=strict")
	}
}

tasks.withType<Test> {
	useJUnitPlatform()
}
