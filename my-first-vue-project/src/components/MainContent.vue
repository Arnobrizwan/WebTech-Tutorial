<template>
    <div class="main">
      <h1>SECJ3483 <br /><span>Web Technology</span> <br />Course</h1>
      <p class="par">
        Lorem ipsum dolor sit amet consectetur adipisicing elit. Sunt neque expedita atque eveniet
        <br />quis nesciunt. Quos nulla vero consequuntur, fugit nemo ad delectus
        <br />a quae totam ipsa illum minus laudantium?
      </p>
      <button class="cn"><a href="#">PARTICIPATE</a></button>
    </div>
  </template>
  
  <script>
  export default {
    name: 'MainContent'
  }
  </script>
  
  <style scoped>
  .main {
    width: 100%;
    height: 100vh;
    background: linear-gradient(
        to top,
        rgba(0, 0, 0, 0.5) 50%,
        rgba(0, 0, 0, 0.5) 50%
      ),
      url('@/assets/image-2.png'); /* ✅ Updated path */
    background-position: center;
    background-size: cover;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    color: white;
    text-align: center;
    padding: 0 20px;
  }
  
  h1 {
    font-size: 55px;
    letter-spacing: 2px;
    font-family: 'Times New Roman', serif;
  }
  
  span {
    color: #ff7200;
    font-size: 65px;
  }
  
  .par {
    padding-bottom: 25px;
    font-family: Arial, sans-serif;
    letter-spacing: 1.2px;
    line-height: 30px;
  }
  
  .cn {
    width: 200px;
    height: 40px;
    background: #ff7200;
    border: none;
    margin-bottom: 10px;
    font-size: 18px;
    border-radius: 10px;
    cursor: pointer;
    transition: 0.4s ease;
  }
  
  .cn a {
    text-decoration: none;
    color: #000;
    display: inline-block;
    width: 100%;
    height: 100%;
    line-height: 40px;
  }
  
  .cn:hover {
    background-color: #fff;
    transform: scale(0.9) translateX(15px) translateY(15px);
  }
  </style>