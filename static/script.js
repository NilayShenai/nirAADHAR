document.addEventListener("DOMContentLoaded", () => {
    const uploadSection = document.getElementById("upload-section");
    const editorSection = document.getElementById("editor-section");
    const resultSection = document.getElementById("result-section");
    const fileInput = document.getElementById("file-input");
    const dropZone = document.getElementById("drop-zone");
    const imagePreview = document.getElementById("image-preview");
    const resetButton = document.getElementById("reset-button");
    const digitBtns = document.querySelectorAll(".digit-btn");
    const processButton = document.getElementById("process-button");
    const spinner = document.getElementById("spinner");
    const btnText = processButton.querySelector(".btn-text");
    const imageResult = document.getElementById("image-result");
    const downloadButton = document.getElementById("download-button");
    const copyButton = document.getElementById("copy-button");
    const restartButton = document.getElementById("restart-button");
    let uploadedFile = null;
    let selectedDigit = null;
    let processedImageBlob = null;
    dropZone.addEventListener("click", () => {
        fileInput.click();
    });
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        }, false);
    });
    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        }, false);
    });
    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    window.addEventListener("paste", (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of items) {
            if (item.type.indexOf("image") === 0) {
                const blob = item.getAsFile();
                if (blob) {
                    handleFileSelect(blob);
                    break;
                }
            }
        }
    });
    function handleFileSelect(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file.");
            return;
        }
        uploadedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadSection.classList.add("hidden");
            resultSection.classList.add("hidden");
            editorSection.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }
    function selectDigit(digitStr) {
        digitBtns.forEach(b => {
            if (b.getAttribute("data-digit") === digitStr) {
                b.classList.add("selected");
            } else {
                b.classList.remove("selected");
            }
        });
        selectedDigit = digitStr;
        processButton.disabled = false;
    }
    digitBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            selectDigit(btn.getAttribute("data-digit"));
        });
    });
    window.addEventListener("keydown", (e) => {
        if (!editorSection.classList.contains("hidden")) {
            if (e.key >= '0' && e.key <= '9') {
                selectDigit(e.key);
            } else if (e.key === "Enter" && !processButton.disabled) {
                processButton.click();
            } else if (e.key === "Escape") {
                resetToUpload();
            }
        } else if (!resultSection.classList.contains("hidden")) {
            if (e.key === "Escape" || e.key === "Backspace") {
                resetToUpload();
            }
        }
    });
    resetButton.addEventListener("click", resetToUpload);
    restartButton.addEventListener("click", resetToUpload);
    function resetToUpload() {
        uploadedFile = null;
        selectedDigit = null;
        processedImageBlob = null;
        fileInput.value = "";
        imagePreview.src = "";
        imageResult.src = "";
        digitBtns.forEach(b => b.classList.remove("selected"));
        processButton.disabled = true;
        uploadSection.classList.remove("hidden");
        editorSection.classList.add("hidden");
        resultSection.classList.add("hidden");
    }
    processButton.addEventListener("click", async () => {
        if (!uploadedFile || selectedDigit === null) return;
        processButton.disabled = true;
        spinner.classList.remove("hidden");
        btnText.textContent = "[ processing... ]";
        const formData = new FormData();
        formData.append("file", uploadedFile);
        formData.append("digit", selectedDigit);
        formData.append("platform", "auto");
        try {
            const response = await fetch("/api/modify", {
                method: "POST",
                body: formData
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Error processing image");
            }
            processedImageBlob = await response.blob();
            const resultUrl = URL.createObjectURL(processedImageBlob);
            imageResult.src = resultUrl;
            editorSection.classList.add("hidden");
            resultSection.classList.remove("hidden");
        } catch (error) {
            console.error(error);
            alert(`Processing failed: ${error.message}`);
        } finally {
            processButton.disabled = false;
            spinner.classList.add("hidden");
            btnText.textContent = "[ process image ]";
        }
    });
    downloadButton.addEventListener("click", () => {
        if (!processedImageBlob) return;
        const link = document.createElement("a");
        link.href = URL.createObjectURL(processedImageBlob);
        const originalName = (uploadedFile.name || "card").split('.').slice(0, -1).join('.') || "card";
        link.download = `${originalName}_modified.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
    if (copyButton) {
        copyButton.addEventListener("click", async () => {
            if (!processedImageBlob) return;
            try {
                const item = new ClipboardItem({ "image/png": processedImageBlob });
                await navigator.clipboard.write([item]);
                const origText = copyButton.textContent;
                copyButton.textContent = "[ copied! ]";
                setTimeout(() => {
                    copyButton.textContent = origText;
                }, 2000);
            } catch (err) {
                console.error("Clipboard copy failed:", err);
                alert("Direct clipboard copy is not supported in this browser. Please use download.");
            }
        });
    }
});