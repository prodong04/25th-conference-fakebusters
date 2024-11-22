import Image from 'next/image'
import Link from 'next/link'
import styles from './Header.module.css'

const Header = () => {
  return (
    <header className={styles.header}>
      <div className={styles.logoContainer}>
        <div className={styles.logo}>
          <Image 
            src="/icon.png" 
            alt="Fakebusters Logo" 
            fill
            style={{ objectFit: 'cover' }}
          />
        </div>
        <h1 className={styles.title}>Deepfake Detector</h1>
      </div>
      <nav className={styles.nav}>
        <Link href="/" className={styles.navLink}>Home</Link>
        <Link href="https://github.com/sunghj1118/25th-conference-fakebusters" className={styles.navLink}>Github</Link>
        <Link href="/docs" className={styles.navLink}>Documentation</Link>
      </nav>
    </header>
  )
}

export default Header