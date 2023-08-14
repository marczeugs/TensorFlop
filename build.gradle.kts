plugins {
    kotlin("jvm") version "1.9.0"
    kotlin("plugin.serialization") version "1.9.0"
}

group = "marczeugs"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains.kotlinx:multik-core:0.2.2")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.2")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0-RC")
    implementation("org.knowm.xchart:xchart:3.8.5")

    testImplementation("io.kotest:kotest-runner-junit5:5.6.2")
    testImplementation("io.kotest:kotest-assertions-core:5.6.2")
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(17)
}