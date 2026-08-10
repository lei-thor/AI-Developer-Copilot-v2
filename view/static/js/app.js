(function () {
  "use strict";

  var UPLOAD_API = "/api/knowledge/files/upload";
  var DOCUMENTS_API = "/api/knowledge/files";
  var LOCAL_DOCUMENTS_KEY = "ai_developer_copilot_recent_documents";
  var documentsState = [];

  function showToast(message) {
    var toast = document.querySelector("[data-toast]");
    if (!toast) {
      return;
    }
    toast.textContent = message;
    toast.classList.add("visible");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(function () {
      toast.classList.remove("visible");
    }, 2600);
  }

  function initializeNavigation() {
    var page = document.body.dataset.page;
    document.querySelectorAll("[data-nav]").forEach(function (link) {
      if (link.dataset.nav === page) {
        link.classList.add("active");
      }
    });

    var menuButton = document.querySelector("[data-menu-toggle]");
    var sidebar = document.querySelector(".sidebar");
    if (menuButton && sidebar) {
      menuButton.addEventListener("click", function () {
        sidebar.classList.toggle("open");
      });
    }
  }

  function initializePlaceholderActions() {
    document.querySelectorAll("[data-placeholder-action]").forEach(function (button) {
      button.addEventListener("click", function () {
        showToast(button.dataset.placeholderAction || "该功能将在下一阶段接入。");
      });
    });
  }

  function initializeChatPlaceholder() {
    var form = document.querySelector("[data-chat-form]");
    var input = document.querySelector("[data-chat-input]");
    var messages = document.querySelector("[data-chat-messages]");
    if (!form || !input || !messages) {
      return;
    }

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var text = input.value.trim();
      if (!text) {
        showToast("请输入问题。");
        return;
      }

      var welcome = messages.querySelector(".chat-welcome");
      if (welcome) {
        welcome.remove();
      }

      var userMessage = document.createElement("div");
      userMessage.className = "chat-message user";
      userMessage.innerHTML =
        '<div class="message-avatar">我</div><div class="message-body"></div>';
      userMessage.querySelector(".message-body").textContent = text;
      messages.appendChild(userMessage);

      var assistantMessage = document.createElement("div");
      assistantMessage.className = "chat-message";
      assistantMessage.innerHTML =
        '<div class="message-avatar">助</div><div class="message-body"></div>';
      assistantMessage.querySelector(".message-body").textContent =
        "聊天接口已预留，当前页面暂不发送真实请求。";
      messages.appendChild(assistantMessage);
      input.value = "";
      messages.scrollTop = messages.scrollHeight;
    });
  }

  function initializeDocumentsPage() {
    if (document.body.dataset.page !== "documents") {
      return;
    }

    var uploadZone = document.querySelector("[data-upload-zone]");
    var fileInput = document.querySelector("[data-file-input]");
    var uploadButton = document.querySelector("[data-upload-button]");
    var refreshButton = document.querySelector("[data-refresh-documents]");

    documentsState = readLocalDocuments();
    renderDocuments(documentsState);
    loadDocuments(false);

    if (uploadButton && fileInput) {
      uploadButton.addEventListener("click", function () {
        fileInput.click();
      });
    }

    if (fileInput) {
      fileInput.addEventListener("change", function () {
        handleFiles(fileInput.files);
        fileInput.value = "";
      });
    }

    if (uploadZone) {
      ["dragenter", "dragover"].forEach(function (eventName) {
        uploadZone.addEventListener(eventName, function (event) {
          event.preventDefault();
          uploadZone.classList.add("drag-over");
        });
      });

      ["dragleave", "drop"].forEach(function (eventName) {
        uploadZone.addEventListener(eventName, function (event) {
          event.preventDefault();
          uploadZone.classList.remove("drag-over");
        });
      });

      uploadZone.addEventListener("drop", function (event) {
        handleFiles(event.dataTransfer.files);
      });
    }

    if (refreshButton) {
      refreshButton.addEventListener("click", function () {
        loadDocuments(true);
      });
    }
  }

  async function handleFiles(fileList) {
    var files = Array.prototype.slice.call(fileList || []);
    if (!files.length) {
      return;
    }

    for (var index = 0; index < files.length; index += 1) {
      await uploadFile(files[index]);
    }
  }

  async function uploadFile(file) {
    if (!isSupportedFile(file.name)) {
      setUploadStatus("失败：仅支持 PDF、Word、Markdown。", "failed");
      showToast("仅支持 PDF、Word、Markdown 文件。");
      return;
    }

    var localId = "local-" + Date.now() + "-" + Math.random().toString(16).slice(2);
    var record = {
      id: localId,
      fileName: file.name,
      fileType: fileTypeFromName(file.name),
      fileSize: file.size,
      uploadStatus: "uploading",
      uploadTime: new Date().toISOString(),
      message: "正在上传...",
    };

    upsertDocument(record);
    setUploadStatus("正在上传...", "uploading");

    var processingTimer = window.setTimeout(function () {
      record.uploadStatus = "processing";
      record.message = "文档解析中...";
      upsertDocument(record);
      setUploadStatus("文档解析中...", "processing");
    }, 900);

    try {
      var formData = new FormData();
      formData.append("file", file);

      var response = await fetch(UPLOAD_API, {
        method: "POST",
        body: formData,
      });
      var payload = await readJsonResponse(response);
      if (!response.ok || payload.success === false) {
        throw new Error(payload.message || "文件上传失败。");
      }

      window.clearTimeout(processingTimer);
      var uploaded = normalizeUploadedDocument(payload, file, localId);
      removeDocument(localId);
      upsertDocument(uploaded);
      saveLocalDocuments(documentsState);
      setUploadStatus("已完成", "completed");
      showToast("文档上传入库完成。");
    } catch (error) {
      window.clearTimeout(processingTimer);
      record.uploadStatus = "failed";
      record.message = error.message || "失败";
      upsertDocument(record);
      saveLocalDocuments(documentsState);
      setUploadStatus("失败：" + record.message, "failed");
      showToast(record.message);
    }
  }

  async function loadDocuments(showMessage) {
    var localDocuments = readLocalDocuments();
    if (localDocuments.length) {
      documentsState = localDocuments;
      renderDocuments(documentsState);
    }

    try {
      var response = await fetch(DOCUMENTS_API, { method: "GET" });
      if (!response.ok) {
        throw new Error("文档列表接口暂未接入。");
      }
      var payload = await response.json();
      var documents = normalizeDocumentList(payload);
      documentsState = documents;
      renderDocuments(documentsState);
      if (showMessage) {
        showToast("文档列表已刷新。");
      }
    } catch (error) {
      documentsState = localDocuments;
      renderDocuments(documentsState);
      if (showMessage) {
        showToast("服务端文档列表接口暂未接入，已显示本地最近上传记录。");
      }
    }
  }

  function renderDocuments(documents) {
    var list = document.querySelector("[data-document-list]");
    var count = document.querySelector("[data-document-count]");
    if (!list) {
      return;
    }

    list.innerHTML = "";
    if (count) {
      count.textContent = documents.length ? documents.length + " 个文档" : "暂无文档";
    }

    if (!documents.length) {
      var empty = document.createElement("div");
      empty.className = "document-empty";
      empty.innerHTML =
        '<div><div class="empty-icon">文</div><h3>暂无文档记录</h3><p>上传 PDF、Word 或 Markdown 后，这里会展示文件名称、类型、大小、状态和上传时间。服务端列表接口接入后将自动读取真实文档列表。</p></div>';
      list.appendChild(empty);
      return;
    }

    documents.forEach(function (documentRecord) {
      list.appendChild(createDocumentCard(documentRecord));
    });
  }

  function createDocumentCard(documentRecord) {
    var status = statusInfo(documentRecord.uploadStatus);
    var card = document.createElement("article");
    card.className = "document-card";

    var head = document.createElement("div");
    head.className = "document-card-head";

    var titleWrap = document.createElement("div");
    titleWrap.className = "document-title-wrap";

    var typeBadge = document.createElement("span");
    typeBadge.className = "file-type";
    typeBadge.textContent = shortFileType(documentRecord.fileType);

    var titleBlock = document.createElement("div");
    titleBlock.style.minWidth = "0";

    var title = document.createElement("div");
    title.className = "document-title";
    title.textContent = documentRecord.fileName || "未命名文档";

    var summary = document.createElement("div");
    summary.className = "file-meta";
    summary.textContent = documentRecord.message || status.label;

    titleBlock.appendChild(title);
    titleBlock.appendChild(summary);
    titleWrap.appendChild(typeBadge);
    titleWrap.appendChild(titleBlock);

    var statusBadge = document.createElement("span");
    statusBadge.className = "document-status " + status.className;
    statusBadge.textContent = status.label;

    head.appendChild(titleWrap);
    head.appendChild(statusBadge);
    card.appendChild(head);

    var grid = document.createElement("div");
    grid.className = "document-meta-grid";
    [
      ["文件类型", documentRecord.fileType || "-"],
      ["文件大小", formatFileSize(documentRecord.fileSize)],
      ["上传状态", status.label],
      ["上传时间", formatDateTime(documentRecord.uploadTime)],
    ].forEach(function (item) {
      grid.appendChild(createDocumentMeta(item[0], item[1]));
    });

    card.appendChild(grid);
    return card;
  }

  function createDocumentMeta(label, value) {
    var box = document.createElement("div");
    box.className = "document-meta";

    var labelElement = document.createElement("div");
    labelElement.className = "document-meta-label";
    labelElement.textContent = label;

    var valueElement = document.createElement("div");
    valueElement.className = "document-meta-value";
    valueElement.textContent = value || "-";

    box.appendChild(labelElement);
    box.appendChild(valueElement);
    return box;
  }

  function upsertDocument(documentRecord) {
    var existingIndex = documentsState.findIndex(function (item) {
      return item.id === documentRecord.id;
    });
    if (existingIndex >= 0) {
      documentsState.splice(existingIndex, 1, documentRecord);
    } else {
      documentsState.unshift(documentRecord);
    }
    renderDocuments(documentsState);
  }

  function removeDocument(documentId) {
    documentsState = documentsState.filter(function (item) {
      return item.id !== documentId;
    });
  }

  function normalizeUploadedDocument(payload, file, fallbackId) {
    var ingestion = payload.ingestion || {};
    var chunksCount = ingestion.chunks_count || ingestion.inserted_count || "";
    return {
      id: payload.file_id || fallbackId,
      fileName: payload.filename || file.name,
      fileType: displayFileType(payload.file_type || fileTypeFromName(file.name)),
      fileSize: payload.file_size || file.size,
      uploadStatus: "completed",
      uploadTime: new Date().toISOString(),
      message: chunksCount ? "已完成，生成 " + chunksCount + " 个知识片段" : "已完成",
    };
  }

  function normalizeDocumentList(payload) {
    var items = [];
    if (Array.isArray(payload)) {
      items = payload;
    } else if (payload && Array.isArray(payload.files)) {
      items = payload.files;
    } else if (payload && Array.isArray(payload.documents)) {
      items = payload.documents;
    } else if (payload && Array.isArray(payload.results)) {
      items = payload.results;
    }

    return items.map(function (item, index) {
      return {
        id: item.file_id || item.id || "server-" + index,
        fileName: item.filename || item.file_name || item.name || "未命名文档",
        fileType: displayFileType(item.file_type || item.type || item.filename || item.file_name),
        fileSize: item.file_size || item.size || 0,
        uploadStatus: normalizeStatus(item.status || item.ingestion_status || "completed"),
        uploadTime: item.upload_time || item.created_at || item.create_time || "",
        message: item.message || "",
      };
    });
  }

  function readLocalDocuments() {
    try {
      var raw = window.localStorage.getItem(LOCAL_DOCUMENTS_KEY);
      var parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      return [];
    }
  }

  function saveLocalDocuments(documents) {
    try {
      window.localStorage.setItem(
        LOCAL_DOCUMENTS_KEY,
        JSON.stringify(documents.slice(0, 30))
      );
    } catch (error) {
      return;
    }
  }

  function setUploadStatus(message, className) {
    var status = document.querySelector("[data-upload-status]");
    if (!status) {
      return;
    }
    status.className = "upload-status " + (className || "waiting");
    status.textContent = message || "等待上传";
  }

  async function readJsonResponse(response) {
    try {
      return await response.json();
    } catch (error) {
      return { success: false, message: "服务端返回内容无法解析。" };
    }
  }

  function isSupportedFile(fileName) {
    return [".pdf", ".docx", ".md", ".markdown"].indexOf(fileExtension(fileName)) >= 0;
  }

  function fileExtension(fileName) {
    var lower = String(fileName || "").toLowerCase();
    var dotIndex = lower.lastIndexOf(".");
    return dotIndex >= 0 ? lower.slice(dotIndex) : "";
  }

  function fileTypeFromName(fileName) {
    var extension = fileExtension(fileName);
    if (extension === ".pdf") {
      return "PDF";
    }
    if (extension === ".docx") {
      return "Word";
    }
    if (extension === ".md" || extension === ".markdown") {
      return "Markdown";
    }
    return "未知";
  }

  function displayFileType(value) {
    var text = String(value || "").toLowerCase();
    if (text.indexOf("pdf") >= 0 || text.endsWith(".pdf")) {
      return "PDF";
    }
    if (text.indexOf("word") >= 0 || text.indexOf("docx") >= 0 || text.endsWith(".docx")) {
      return "Word";
    }
    if (text.indexOf("markdown") >= 0 || text === "md" || text.endsWith(".md")) {
      return "Markdown";
    }
    return value || "未知";
  }

  function shortFileType(value) {
    if (value === "Markdown") {
      return "MD";
    }
    if (value === "Word") {
      return "DOCX";
    }
    return value || "-";
  }

  function normalizeStatus(value) {
    var text = String(value || "").toLowerCase();
    if (["uploaded", "indexed", "completed", "success", "done"].indexOf(text) >= 0) {
      return "completed";
    }
    if (["processing", "parsing", "embedding"].indexOf(text) >= 0) {
      return "processing";
    }
    if (["uploading"].indexOf(text) >= 0) {
      return "uploading";
    }
    if (["failed", "error"].indexOf(text) >= 0) {
      return "failed";
    }
    return "waiting";
  }

  function statusInfo(status) {
    var map = {
      waiting: { label: "等待上传", className: "waiting" },
      uploading: { label: "正在上传...", className: "uploading" },
      processing: { label: "文档解析中...", className: "processing" },
      completed: { label: "已完成", className: "completed" },
      failed: { label: "失败", className: "failed" },
    };
    return map[status] || map.waiting;
  }

  function formatFileSize(size) {
    var value = Number(size);
    if (!value || value < 0) {
      return "-";
    }
    var units = ["B", "KB", "MB", "GB"];
    var unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex += 1;
    }
    return (unitIndex === 0 ? value : value.toFixed(1)) + units[unitIndex];
  }

  function formatDateTime(value) {
    if (!value) {
      return "-";
    }
    var date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return String(value);
    }
    return date.toLocaleString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initializeNavigation();
    initializePlaceholderActions();
    initializeChatPlaceholder();
    initializeDocumentsPage();
  });
})();
