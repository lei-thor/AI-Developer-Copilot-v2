(function () {
  "use strict";

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

  document.addEventListener("DOMContentLoaded", function () {
    initializeNavigation();
    initializePlaceholderActions();
    initializeChatPlaceholder();
  });
})();
