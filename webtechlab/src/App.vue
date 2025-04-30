<template>
  <div class="main">
    <AppNavbar />
    <MainContent />
    <AppFooter />
  </div>
</template>

<script>
import AppNavbar from './components/Navbar.vue'
import MainContent from './components/MainContent.vue'
import AppFooter from './components/Footer.vue'

export default {
  name: 'App',
  components: {
    AppNavbar,
    MainContent,
    AppFooter
  }
}
</script>

<style>
html, body {
  margin: 0;
  padding: 0;
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  background-color: #f5f5f5;
}
</style>