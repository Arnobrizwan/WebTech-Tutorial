<template>
    <div class="navbar">
      <div class="icon">
        <img src="@/assets/image-2.png" alt="Vue logo" class="logo" />
      </div>
      <div class="menu">
        <ul>
          <li><a href="#">HOME</a></li>
          <li><a href="#">ABOUT</a></li>
          <li><a href="#">COURSE MATERIALS</a></li>
          <li><a href="#">ASSESSMENT</a></li>
          <li><a href="#">CONTACT US</a></li>
        </ul>
      </div>
      <div class="search">
        <input class="srch" type="search" placeholder="Type To Find" />
        <button class="btn">Find</button>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'AppNavbar'
  }
  </script>
  
  <style scoped>
  .navbar {
    width: 100%;
    height: 75px;
    background-color: rgba(0, 0, 0, 0.9);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 30px;
    color: white;
    position: fixed;
    top: 0;
    left: 0;
    z-index: 1000;
  }
  
  .icon {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    overflow: hidden;
    background-color: white;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .logo {
    width: 80%;
    height: 80%;
    object-fit: cover;
  }
  
  .menu ul {
    display: flex;
    list-style: none;
    gap: 30px;
  }
  
  .menu ul li a {
    color: #fff;
    text-decoration: none;
    font-weight: bold;
    font-family: Arial, sans-serif;
    transition: 0.3s ease;
  }
  
  .menu ul li a:hover {
    color: #ff7200;
  }
  
  .search {
    display: flex;
    align-items: center;
  }
  
  .srch {
    width: 140px;
    height: 30px;
    background: transparent;
    border: 1px solid #ff7200;
    color: #fff;
    font-size: 14px;
    padding: 5px 10px;
    border-radius: 5px 0 0 5px;
  }
  
  .btn {
    width: 60px;
    height: 30px;
    background: #ff7200;
    color: white;
    border: 1px solid #ff7200;
    font-size: 14px;
    border-radius: 0 5px 5px 0;
    cursor: pointer;
    transition: 0.3s;
  }
  
  .btn:hover {
    background: #fff;
    color: #000;
  }
  </style>